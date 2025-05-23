import re
import unittest
import numpy as np
import discretize
import pytest

from simpeg import maps, simulation


class TestLinearSimulation(unittest.TestCase):
    def setUp(self):
        mesh = discretize.TensorMesh([100])
        self.sim = simulation.ExponentialSinusoidSimulation(
            mesh=mesh,
            model_map=maps.IdentityMap(mesh),
        )

        mtrue = np.zeros(mesh.nC)
        mtrue[mesh.cell_centers_x > 0.3] = 1.0
        mtrue[mesh.cell_centers_x > 0.45] = -0.5
        mtrue[mesh.cell_centers_x > 0.6] = 0

        self.mtrue = mtrue

    def test_make_synthetic_data(self):
        dclean = self.sim.dpred(self.mtrue)
        data = self.sim.make_synthetic_data(self.mtrue)
        assert np.all(data.relative_error == 0.05 * np.ones_like(dclean))


def test_exp_sim_mesh_required():
    msg = ".*missing 1 required positional argument: 'mesh'"
    with pytest.raises(TypeError, match=msg):
        simulation.ExponentialSinusoidSimulation()


def test_exp_sim_bad_mesh_type():
    mesh = discretize.TreeMesh([16, 16])
    msg = "mesh must be an instance of TensorMesh, not TreeMesh"
    with pytest.raises(TypeError, match=msg):
        simulation.ExponentialSinusoidSimulation(mesh)


def test_exp_sim_bad_mesh_dim():
    mesh = discretize.TensorMesh([5, 5])
    msg = "ExponentialSinusoidSimulation mesh must be 1D, received a 2D mesh."
    with pytest.raises(ValueError, match=msg):
        simulation.ExponentialSinusoidSimulation(mesh)


@pytest.mark.parametrize(
    "base_class",
    [
        simulation.BaseSimulation,
        simulation.BaseTimeSimulation,
        simulation.LinearSimulation,
    ],
)
def test_base_no_mesh(base_class):
    # this current message is... not a good one
    # # but is what is thrown when an invalid arugment is passed.
    msg = re.escape(
        "object.__init__() takes exactly one argument (the instance to initialize)"
    )
    with pytest.raises(TypeError, match=msg):
        base_class(mesh=0)


class TestTimeSimulation(unittest.TestCase):
    def setUp(self):
        self.sim = simulation.BaseTimeSimulation()

    def test_time_simulation_time_steps(self):
        self.sim.time_steps = [(1e-6, 3), 1e-5, (1e-4, 2)]
        true_time_steps = np.r_[1e-6, 1e-6, 1e-6, 1e-5, 1e-4, 1e-4]
        self.assertTrue(np.all(true_time_steps == self.sim.time_steps))

        true_time_steps = np.r_[1e-7, 1e-6, 1e-6, 1e-5, 1e-4, 1e-4]
        self.sim.time_steps = true_time_steps
        self.assertTrue(np.all(true_time_steps == self.sim.time_steps))

        self.assertTrue(self.sim.nT == 6)

        self.assertTrue(np.all(self.sim.times == np.r_[0, true_time_steps].cumsum()))

        self.sim.t0 = 1
        self.assertTrue(np.all(self.sim.times == np.r_[1, true_time_steps].cumsum()))


if __name__ == "__main__":
    unittest.main()
