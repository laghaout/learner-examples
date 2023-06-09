import os
import pytest
from learners import utilities as util
from learners import learner as lea


@pytest.fixture
def learner():
    # Determine the environment parameters and the corresponding paths.
    env = util.EnvManager(
        container_params=("INSIDE_DOCKER_CONTAINER", "INSIDE_GCP")
    )
    env(
        **dict(
            lesson_dir="lesson",  # Directory where to store the lesson.
            dir_name="iris",  # Name of the current directory.
        )
    )

    kwargs = dict(
        lesson_dir=env.paths.lesson_dir,
        data_params=dict(
            data_source=os.path.join(
                env.paths.data_dir, f"{env.paths.dir_name}.csv"
            ),
            nrows=None,
            delimiter="\t",
            shuffle_seed=41,
            parts={"train": 0.8, "test": 0.2},
            fillna=0,
            target_name="species",
            features=[
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
            ],
        ),
        hyperparams=dict(
            early_stopping_threshold=0.9999,
            epochs=145,  # 100
            batch_size=16,
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
            hidden_units=[16],
            output_units=3,
            activation="relu",
            output_activation="softmax",
        ),
        env=env,
    )

    learner = lea.Learner(**kwargs)
    learner(explore=True, select=False, train=True, test=True, serve=False)

    return learner


# %%


def test_learner_has_report(learner) -> None:
    """Test that the learner object has a report attribute."""

    assert hasattr(learner, "report")


def test_learner_has_data(learner) -> None:
    """Test that the learner object has a data attribute."""

    assert hasattr(learner, "data")


def test_something_else(learner) -> None:
    """Test that we raise an error under some condition."""

    with pytest.raises(Exception):
        if True:
            raise Exception("Running some error code.")
