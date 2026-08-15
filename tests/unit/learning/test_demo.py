"""Tests for the synthetic learning demonstration."""

from robot.learning.demo import demo_regression, demo_xor


class TestDemoXOR:
    def test_loss_decreases(self) -> None:
        result = demo_xor()
        assert result["loss_decreased"], (
            f"XOR loss did not decrease: initial={result['initial_loss']:.6f}, "
            f"final={result['final_loss']:.6f}"
        )

    def test_predictions_improve(self) -> None:
        result = demo_xor()
        # After training, XOR predictions should be close to targets
        pred = result["final_pred"]
        result["target"]
        # XOR(0,0)≈0, XOR(0,1)≈1, XOR(1,0)≈1, XOR(1,1)≈0
        assert pred[0] < 0.3, f"XOR(0,0)={pred[0]:.4f}, expected <0.3"  # type: ignore[index]
        assert pred[1] > 0.7, f"XOR(0,1)={pred[1]:.4f}, expected >0.7"  # type: ignore[index]
        assert pred[2] > 0.7, f"XOR(1,0)={pred[2]:.4f}, expected >0.7"  # type: ignore[index]
        assert pred[3] < 0.3, f"XOR(1,1)={pred[3]:.4f}, expected <0.3"  # type: ignore[index]

    def test_model_is_small(self) -> None:
        result = demo_xor()
        # MLP with [8,8] hidden, 2 inputs, 1 output should be <200 params
        assert result["param_count"] < 200  # type: ignore[operator]

    def test_final_loss_is_low(self) -> None:
        result = demo_xor()
        assert result["final_loss"] < 0.1, f"XOR final loss too high: {result['final_loss']:.6f}"  # type: ignore[operator]


class TestDemoRegression:
    def test_loss_decreases(self) -> None:
        result = demo_regression()
        assert result["loss_decreased"], (
            f"Regression loss did not decrease: initial={result['initial_loss']:.6f}, "
            f"final={result['final_loss']:.6f}"
        )

    def test_prediction_close_to_expected(self) -> None:
        result = demo_regression()
        # Test input is 0.5, expected output ≈ 2*0.5 + 1 = 2.0
        pred = result["test_prediction"]
        expected = result["expected_approx"]
        assert abs(pred - expected) < 0.5, (  # type: ignore[operator]
            f"Prediction {pred:.4f} too far from expected {expected:.4f}"
        )

    def test_model_is_small(self) -> None:
        result = demo_regression()
        assert result["param_count"] < 500  # type: ignore[operator]
