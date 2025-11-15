import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from drug_discovery_simulation import (
    ActiveLearningCoordinator,
    DatasetManager,
    FeatureExtractor,
    LigandDiscoveryAgent,
    MLInferenceAPI,
    MLModelRegistry,
    PhysicalValidator,
    PublicDataClient,
    QuantumBlackboard,
    QuantumContext,
    QuantumSimulationAgent,
    ScreeningAgent,
)
from rl_rewards import RewardPrimitives


class StubLLM:
    def complete(self, prompt: str, temperature: float = 0.2) -> str:  # pragma: no cover - deterministic stub
        return "Stub completion"


class DummyDataClient(PublicDataClient):
    def fetch_pubchem_candidates(self, query: str):  # pragma: no cover - deterministic stub
        return [
            {"ligandId": "cand-1", "smiles": "CCO"},
            {"ligandId": "cand-2", "smiles": "CCC"},
        ]

    def fetch_patent_hits(self, query: str):  # pragma: no cover - deterministic stub
        return ["US-123456", "EP-987654"]

    def fetch_uniprot_metadata(self, accession: str) -> Dict[str, Any]:  # pragma: no cover - deterministic stub
        return {"accession": accession, "recommendedName": "Stub Target"}


@pytest.fixture
def simple_context() -> QuantumContext:
    descriptors = [
        {
            "shellIndex": 0,
            "lambdaRadius": 1.0,
            "lambdaCurvature": 0.1,
            "lambdaEntropy": 0.3,
            "lambdaEnergyDensity": -0.5,
            "lambdaBhattacharyya": 0.6,
            "lambdaOccupancy": 0.5,
            "lambdaLeakage": 0.1,
        },
        {
            "shellIndex": 1,
            "lambdaRadius": 1.2,
            "lambdaCurvature": 0.2,
            "lambdaEntropy": 0.4,
            "lambdaEnergyDensity": -0.4,
            "lambdaBhattacharyya": 0.5,
            "lambdaOccupancy": 0.4,
            "lambdaLeakage": 0.1,
        },
    ]
    return QuantumContext(
        curvature_profile=[0.1, 0.2],
        lambda_modes=[1.0, 1.1],
        entanglement_entropy=0.5,
        enhancement_factor=0.3,
        lambda_shells=descriptors,
        lambda_basis={"basis": "lambda"},
    )


@pytest.fixture
def quantum_reference(simple_context: QuantumContext) -> Dict[str, Any]:
    return {
        "samples": [
            {
                "ligandId": "ref-1",
                "bindingEnergy": -8.2,
                "entanglementEntropy": 0.45,
                "orbitalOccupations": [0.25, 0.25, 0.25, 0.25],
            }
        ],
        "statistics": {"meanEnergy": -8.0, "energyRange": {"min": -20.0, "max": -3.0}},
    }


@pytest.fixture
def shared_components(simple_context: QuantumContext, quantum_reference: Dict[str, Any]):
    feature_extractor = FeatureExtractor()
    dataset_manager = DatasetManager(feature_extractor)
    ml_registry = MLModelRegistry()
    active_learning = ActiveLearningCoordinator()
    ml_api = MLInferenceAPI()
    validator = PhysicalValidator(simple_context, quantum_reference)
    return {
        "feature_extractor": feature_extractor,
        "dataset_manager": dataset_manager,
        "ml_registry": ml_registry,
        "active_learning": active_learning,
        "ml_api": ml_api,
        "validator": validator,
    }


async def run_ligand_agent(simple_context, quantum_reference, shared_components, blackboard):
    data_client = DummyDataClient()
    llm = StubLLM()
    agent = LigandDiscoveryAgent(
        blackboard,
        simple_context,
        shared_components["validator"],
        data_client,
        llm,
        target_query="aspirin",
        quantum_reference=quantum_reference,
        ml_registry=shared_components["ml_registry"],
        dataset_manager=shared_components["dataset_manager"],
        feature_extractor=shared_components["feature_extractor"],
        active_learning=shared_components["active_learning"],
        ml_api=shared_components["ml_api"],
    )
    binding_seed = {"pockets": [{"pocketId": "pocket-01"}]}
    await blackboard.post("binding", binding_seed)
    seed_ligands = [
        {"ligandId": "seed-1", "smiles": "CCO", "noveltyScore": 0.4, "syntheticSteps": 2},
        {"ligandId": "seed-2", "smiles": "CCC", "noveltyScore": 0.5, "syntheticSteps": 2},
    ]
    return await agent.run(simple_context, seed_ligands, beam_width=4)


def test_ligand_agent_filters_and_reports_counts(simple_context, quantum_reference, shared_components):
    blackboard = QuantumBlackboard()
    report = asyncio.run(run_ligand_agent(simple_context, quantum_reference, shared_components, blackboard))
    total = len(report.get("generatedLigands", [])) + len(report.get("fallbackLigands", []))
    stats = report["stats"]
    assert stats["acceptedCount"] + stats["rejectedCount"] == total
    assert len(report["generatedLigands"]) == stats["acceptedCount"]


def test_no_null_ligand_ids(simple_context, quantum_reference, shared_components):
    blackboard = QuantumBlackboard()
    report = asyncio.run(run_ligand_agent(simple_context, quantum_reference, shared_components, blackboard))
    for ligand in report["generatedLigands"]:
        assert ligand.get("ligandId")
        assert ligand.get("smiles")


def test_quantum_agent_energy_within_range_on_fixture(simple_context, quantum_reference, shared_components):
    blackboard = QuantumBlackboard()
    asyncio.run(run_ligand_agent(simple_context, quantum_reference, shared_components, blackboard))
    data_client = DummyDataClient()
    uniprot_meta = data_client.fetch_uniprot_metadata("P12345")
    agent = QuantumSimulationAgent(
        blackboard,
        simple_context,
        shared_components["validator"],
        uniprot_meta,
        quantum_reference,
        ml_registry=shared_components["ml_registry"],
        dataset_manager=shared_components["dataset_manager"],
        feature_extractor=shared_components["feature_extractor"],
        active_learning=shared_components["active_learning"],
        ml_api=shared_components["ml_api"],
    )
    reports = asyncio.run(agent.run(simple_context))
    energy = reports["best"]["bindingFreeEnergy"]
    assert -25.0 <= energy <= -3.0


def test_physical_binding_energy_range(simple_context, quantum_reference, shared_components):
    blackboard = QuantumBlackboard()
    asyncio.run(run_ligand_agent(simple_context, quantum_reference, shared_components, blackboard))
    agent = QuantumSimulationAgent(
        blackboard,
        simple_context,
        shared_components["validator"],
        {"accession": "P12345"},
        quantum_reference,
        ml_registry=shared_components["ml_registry"],
        dataset_manager=shared_components["dataset_manager"],
        feature_extractor=shared_components["feature_extractor"],
        active_learning=shared_components["active_learning"],
        ml_api=shared_components["ml_api"],
    )
    reports = asyncio.run(agent.run(simple_context))
    hybrid = reports["best"].get("hybridEnergyPipeline", {})
    assert set(["rawQuantumEnergy", "mmffEnergy", "lambdaCorrection"]).issubset(hybrid.keys())


def test_ml_augmentation_present_in_demo(simple_context, quantum_reference, shared_components):
    blackboard = QuantumBlackboard()
    report = asyncio.run(run_ligand_agent(simple_context, quantum_reference, shared_components, blackboard))
    assert "mlAugmentation" in report


def test_agents_log_when_ml_models_missing(simple_context, quantum_reference, shared_components, caplog):
    blackboard = QuantumBlackboard()
    asyncio.run(run_ligand_agent(simple_context, quantum_reference, shared_components, blackboard))
    caplog.set_level("WARNING")
    agent = ScreeningAgent(
        blackboard,
        simple_context,
        shared_components["validator"],
        quantum_reference,
        ml_registry=shared_components["ml_registry"],
        dataset_manager=shared_components["dataset_manager"],
        feature_extractor=shared_components["feature_extractor"],
        active_learning=shared_components["active_learning"],
        ml_api=shared_components["ml_api"],
    )
    asyncio.run(agent.run())
    assert any("model not loaded" in record.message for record in caplog.records)


def test_reward_primitives_basic_bounds():
    assert RewardPrimitives.R_bind(-9.0, -7.0) > 0
    assert RewardPrimitives.R_safety(0.25, 0.1) < 0
    assert RewardPrimitives.R_diversity(0.5) > 0
    flow = RewardPrimitives.R_lambda_flow(np.array([0.5, 0.4]), np.array([0.5, 0.45]), 0.01)
    assert -1.0 <= flow <= 1.0


def test_ligand_agent_compute_reward(simple_context, quantum_reference, shared_components):
    data_client = DummyDataClient()
    llm = StubLLM()
    agent = LigandDiscoveryAgent(
        QuantumBlackboard(),
        simple_context,
        shared_components["validator"],
        data_client,
        llm,
        target_query="aspirin",
        quantum_reference=quantum_reference,
        ml_registry=shared_components["ml_registry"],
        dataset_manager=shared_components["dataset_manager"],
        feature_extractor=shared_components["feature_extractor"],
        active_learning=shared_components["active_learning"],
        ml_api=shared_components["ml_api"],
    )
    primitives = {
        "binding_energy": -8.4,
        "reference_energy": -7.0,
        "off_target_energy": -7.6,
        "entropy_per_shell": [0.3, 0.4],
        "leakage": [0.1, 0.2],
        "curvature_gradient": 0.05,
        "occupancy": np.array([0.5, 0.45]),
        "occupancy_prev": np.array([0.5, 0.5]),
        "shellEntropyDelta": 0.01,
        "toxicity_prob": 0.2,
        "toxicity_uncertainty": 0.05,
        "ip_distance": 0.6,
        "ip_scale": 0.4,
        "num_accepted": 2,
        "beam_width": 4,
        "diversity_index": 0.5,
        "entropy_mean": 0.35,
        "energy_signal": -8.4,
    }
    reward = agent.compute_reward(primitives)
    assert isinstance(reward, float)
    assert agent.reward_history[-1] == reward
