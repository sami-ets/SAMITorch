# Contributing guidelines

## Pull Request Checklist

Before sending your pull requests, make sure you followed this list.

- Read [contributing guidelines](CONTRIBUTING.md).
- Read [Code of Conduct](CODE_OF_CONDUCT.md).
- Check if my changes are consistent with the [guidelines](https://github.com/sami-ets/SAMITorch/blob/master/CONTRIBUTING.md#general-guidelines-and-philosophy-for-contribution).
- Changes are consistent with the [Coding Style](https://github.com/sami-ets/SAMITorch/blob/master/CONTRIBUTING.md#python-coding-style).
- Run [Unit Tests](https://github.com/sami-ets/SamiTorch/master/CONTRIBUTING.md#running-unit-tests).

## How to become a contributor and submit your own code

### Contributing code

If you have improvements to SAMITorch, send us your pull requests! For those
just getting started, Github has a [howto](https://help.github.com/articles/using-pull-requests/).

SAMITorch team members will be assigned to review your pull requests. Once the
pull requests are approved and pass continuous integration checks, a SAMITorch
team member will apply `ready to pull` label to your change. This means we are
working on getting your pull request submitted to our internal repository. After
the change has been submitted internally, your pull request will be merged
automatically on GitHub.

If you want to contribute but you're not sure where to start, take a look at the
[issues with the "contributions welcome" label](https://github.com/sami-ets/SAMITorch/labels/contributions%20welcome).
These are issues that we believe are particularly well suited for outside
contributions, often because we probably won't get to them right now. If you
decide to start on an issue, leave a comment so that other people know that
you're working on it. If you want to help out, but not alone, use the issue
comment thread to coordinate.

### Contribution guidelines and standards

Before sending your pull request for
[review](https://github.com/sami-ets/SAMITorch/pulls),
make sure your changes are consistent with the guidelines and follow the
SAMITorch coding style.

#### General guidelines and philosophy for contribution

*   Include unit tests when you contribute new features, as they help to a)
    prove that your code works correctly, and b) guard against future breaking
    changes to lower the maintenance cost.
*   Bug fixes also generally require unit tests, because the presence of bugs
    usually indicates insufficient test coverage.
*   Keep API compatibility in mind when you change code in core SAMITorch.
    Reviewers of your pull request will comment on any API compatibility issues.
*   When you contribute a new feature to SAMITorch, the maintenance burden is
    (by default) transferred to the SAMITorch team. This means that benefit of
    the contribution must be compared against the cost of maintaining the
    feature.

#### License

Include a license at the top of new files.

* [Python license example](https://github.com/sami-ets/SAMITorch/blob/master/models/base_model.py#L1)

#### Python coding style

Changes to SAMITorch Python code should conform to
[Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)

Use `pylint` to check your Python changes. To install `pylint` and
retrieve SAMITorch's custom style definition:

```bash
pip install pylint
```

To check a file with `pylint`:

```bash
pylint --rcfile=/tmp/pylintrc myfile.py
```

#### Coding style for other languages

* [Google Java Style Guide](https://google.github.io/styleguide/javaguide.html)
* [Google JavaScript Style Guide](https://google.github.io/styleguide/jsguide.html)
* [Google Shell Style Guide](https://google.github.io/styleguide/shell.xml)
* [Google Objective-C Style Guide](https://google.github.io/styleguide/objcguide.html)

#### Running sanity check

To be described in future.

#### Running unit tests

There are two ways to run SAMITorch unit tests.

1.  Using tools and libraries installed directly on your system.

    Refer to the
    [CPU-only developer Dockerfile](https://github.com/sami-ets/SAMITorch/tree/master/docker/Dockerfile.devel)
    and
    [GPU developer Dockerfile](https://github.com/sami-ets/SAMITorch/tree/master/configs/docker/Dockerfile.devel-gpu)
    for the required packages. Alternatively, Docker images on Docker Hub will be deployed in near future.

    Once you have the packages installed, you can run a specific unit test by doing as follows:

    If the tests are to be run on GPU, add CUDA paths to LD_LIBRARY_PATH and add
    the `cuda` option flag

    ```bash
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
    export flags="--config=opt --config=cuda -k"
    ```

    For example, to run all tests under tests/, do:

    ```python tests/tests.py
    ```

2.  Using [Docker](https://www.docker.com) and SAMITorch's CI scripts.

    ```bash
    # Install Docker first, then this will build and run cpu tests
    python tests/tests.py
    ```
