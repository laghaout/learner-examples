
# Table of Contents



-   [`starter-code`](./starter-code): Starter code for any learner
-   [`ballistics`](./ballistics): Perform forensic ballistics.
-   [`cellular-automata`](./cellular-automata): Predict the behaviour of cellular automata.
-   [`equation-validity`](./equation-validity): Verify the validity of mathematical equations.
-   [`iris`](./iris): Classify flowers.
-   [`quadrant`](./quadrant): Determine the quadrant of a point in `N`-dimensional space.
-   [`sequence`](./sequence): Classify sequences.

-   \`<project>\`
    -   \`Dockerfile\`
    -   \`README.md\`
    -   \`clean.sh\`
    -   \`docker-compose.yml\`
    -   \`devops.sh\`
    -   \`pyproject.toml\`
    -   \`run.sh\`: Alternative to \`docker-compose\`. To be deprecated.
    -   \`.env\`: List fo environment variables
    -   \`<project>/\`
    -   \`<tests>/\`
        -   \`\_<sub>init</sub>\_<sub>.py</sub>\`
        -   \`test\_<some series of tests>.py\`

1.  Run \`sh devops.sh\` to
    -   *test*,
    -   *clean artifacts*, and
    -   create a *pip package* (if \`pip\` is passed as an argument).
2.  To run a particular service specified in the \`docker-compose.yml\`, run \`docker compose run <service>\`.

