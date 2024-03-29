
# Table of Contents

1.  [Directory structure](#org60fa66e)
2.  [Steps](#org6ee9de9)

-   [`starter-code`](./starter-code): Starter code for any learner
-   [`ballistics`](./ballistics): Perform forensic ballistics.
-   [`cellular-automata`](./cellular-automata): Predict the behaviour of cellular automata.
-   [`equation-validity`](./equation-validity): Verify the validity of mathematical equations.
-   [`iris`](./iris): Classify flowers based on iris measurements.
-   [`quadrant`](./quadrant): Determine the quadrant of a point in `N`-dimensional space.
-   [`sequence`](./sequence): Classify sequences.


<a id="org60fa66e"></a>

# Directory structure

-   `<project>`
    -   `Dockerfile`
    -   `README.md`
    -   `clean.sh`
    -   `docker-compose.yml`
    -   `devops.sh`
    -   `pyproject.toml`
    -   `run.sh`: Alternative to `docker-compose`. To be deprecated.
    -   `.env`: List fo environment variables
    -   `<project>/`
    -   `<tests>/`
        -   `__init__.py`
        -   `test_<some series of tests>.py`


<a id="org6ee9de9"></a>

# Steps

1.  Run `sh devops.sh` to
    -   *test*,
    -   *clean artifacts*, and
    -   create a *pip package* (if `pip` is passed as an argument).
2.  To run a particular service specified in the `docker-compose.yml`, run `docker compose run <service>`.

