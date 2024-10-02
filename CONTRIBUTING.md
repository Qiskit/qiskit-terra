# Contributing

Qiskit is an open-source project committed to bringing quantum computing to
people of all backgrounds. This page describes how you can join the Qiskit
community in this goal.


## Contents
* [Before you start](#before-you-start)
* [Choose an issue to work on](#Choose-an-issue-to-work-on)
* [Set up Python virtual development environment](#set-up-python-virtual-development-environment)
* [Installing Qiskit from source](#installing-qiskit-from-source)
* [Issues and pull requests](#issues-and-pull-requests)
* [Contributor Licensing Agreement](#contributor-licensing-agreement)
* [Changelog generation](#changelog-generation)
* [Release notes](#release-notes)
* [Testing](#testing)
* [Style and Lint](#style-and-lint)
* [Building API docs locally](#building-api-docs-locally)
  * [Troubleshooting docs builds](#troubleshooting-docs-builds)
* [Development Cycle](#development-cycle)
  * [Branches](#branches)
  * [Release Cycle](#release-cycle)
* [Adding deprecation warnings](#adding-deprecation-warnings)
* [Using dependencies](#using-dependencies)
  * [Adding a requirement](#adding-a-requirement)
  * [Adding an optional dependency](#adding-an-optional-dependency)
  * [Checking for optionals](#checking-for-optionals)
* [Dealing with git blame ignore list](#dealing-with-the-git-blame-ignore-list)


## Before you start

If you are new to Qiskit contributing we recommend you do the following before diving into the code:

* Read the [Code of Conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md)
* Familiarize yourself with the Qiskit community (via [Slack](https://qisk.it/join-slack),
   [Stack Exchange](https://quantumcomputing.stackexchange.com), [GitHub](https://github.com/qiskit-community/feedback/discussions) etc.)


## Choose an issue to work on
Qiskit uses the following labels to help non-maintainers find issues best suited to their interests and experience level:

* [good first issue](https://github.com/Qiskit/qiskit/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) - these issues are typically the simplest available to work on, ideal for newcomers. They should already be fully scoped, with a clear approach outlined in the descriptions.
* [help wanted](https://github.com/Qiskit/qiskit/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) - these issues are generally more complex than good first issues. They typically cover work that core maintainers don't currently have capacity to implement and may require more investigation/discussion. These are a great option for experienced contributors looking for something a bit more challenging.
* [short project](https://github.com/Qiskit/qiskit/issues?q=is%3Aopen+is%3Aissue+label%3A%22short+project%22) - these issues are bigger pieces of work that require greater time commitment. Good options for hackathons, internship projects etc.


## Set up Python virtual development environment

Virtual environments are used for Qiskit development to isolate the development environment
from system-wide packages. This way, we avoid inadvertently becoming dependent on a
particular system configuration. For developers, this also makes it easy to maintain multiple
environments (e.g. one per supported Python version, for older versions of Qiskit, etc.).

### Set up a Python venv

All Python versions supported by Qiskit include built-in virtual environment module
[venv](https://docs.python.org/3/tutorial/venv.html).

Start by creating a new virtual environment with `venv`. The resulting
environment will use the same version of Python that created it and will not inherit installed
system-wide packages by default. The specified folder will be created and is used to hold the environment's
installation. It can be placed anywhere. For more detail, see the official Python documentation,
[Creation of virtual environments](https://docs.python.org/3/library/venv.html).

```
python3 -m venv ~/.venvs/qiskit-dev
```

Activate the environment by invoking the appropriate activation script for your system, which can
be found within the environment folder. For example, for bash/zsh:


```
source ~/.venvs/qiskit-dev/bin/activate
```

Upgrade pip within the environment to ensure Qiskit dependencies installed in the subsequent sections
can be located for your system.

```
pip install -U pip
```

```
pip install -e .
```

### Set up a Conda environment

For Conda users, a new environment can be created as follows.

```
conda create -y -n QiskitDevenv python=3
conda activate QiskitDevenv
```

```
pip install -e .
```

## Installing Qiskit from source

Qiskit is primarily written in Python but there are some core routines
that are written in the [Rust](https://www.rust-lang.org/) programming
language to improve the runtime performance. For the released versions of
Qiskit we publish precompiled binaries on the
[Python Package Index](https://pypi.org/) for all the supported platforms
which only requires a functional Python environment to install. However, when
building and installing from source you will need a Rust compiler installed. You can do this very easily
using rustup: https://rustup.rs/ which provides a single tool to install and
configure the latest version of the rust compiler.
[Other installation methods](https://forge.rust-lang.org/infra/other-installation-methods.html)
exist too. For Windows users, besides rustup, you will also need install
the Visual C++ build tools so that Rust can link against the system c/c++
libraries. You can see more details on this in the
[rustup documentation](https://rust-lang.github.io/rustup/installation/windows-msvc.html).

If you use Rustup, it will automatically install the correct Rust version
currently used by the project.

Once you have a Rust compiler installed, you can rely on the normal Python
build/install steps to install Qiskit. This means you just run
`pip install .` in your local git clone to build and install Qiskit.

Do note that if you do use develop mode/editable install (via `python setup.py develop` or `pip install -e .`) the Rust extension will be built in debug mode
without any optimizations enabled. This will result in poor runtime performance.
If you'd like to use an editable install with an optimized binary you can
run `python setup.py build_rust --release --inplace` after you install in
editable mode to recompile the rust extensions in release mode.

Note that in order to run `python setup.py ...` commands you need have build
dependency packages installed in your environment, which are listed in the
`pyproject.toml` file under the `[build-system]` section.

### Compile time options

When building qiskit from source there are options available to control how
Qiskit is built. Right now the only option is if you set the environment
variable `QISKIT_NO_CACHE_GATES=1` this will disable runtime caching of
Python gate objects when accessing them from a `QuantumCircuit` or `DAGCircuit`.
This makes a tradeoff between runtime performance for Python access and memory
overhead. Caching gates will result in better runtime for users of Python at
the cost of increased memory consumption. If you're working with any custom
transpiler passes written in Python or are otherwise using a workflow that
repeatedly accesses the `operation` attribute of a `CircuitInstruction` or `op`
attribute of `DAGOpNode` enabling caching is recommended.

## Issues and pull requests

We use [GitHub pull requests](https://help.github.com/articles/about-pull-requests) to accept
contributions.

While not required, opening a new issue about the bug you're fixing or the
feature you're working on before you open a pull request is an important step
in starting a discussion with the community about your work. The issue gives us
a place to talk about the idea and how we can work together to implement it in
the code. It also lets the community know what you're working on, and if you
need help, you can reference the issue when discussing it with other community
and team members.

* For documentation issues relating to pages in the Start, Build, Transpile, Verify, Run, and Migration guides sections of [docs.quantum.ibm.com](https://docs.quantum.ibm.com/), please open an issue in the [Qiskit/documentation repo](https://github.com/Qiskit/documentation/issues/new/choose) rather than the Qiskit/qiskit repo. In other words, any page that DOES NOT have `/api/` in the url should be addressed in the Qiskit/documentation repo.
* For issues relating to API reference pages (any page that contains `/api/` in the url), please open an issue in the repo specific to that API reference, for example [Qiskit/qiskit](https://github.com/Qiskit/qiskit/issues/new/choose), [Qiskit/qiskit-aer](https://github.com/Qiskit/qiskit-aer/issues/new/choose), or [Qiskit/qiskit-ibm-runtime](https://github.com/Qiskit/qiskit-ibm-runtime/issues/new/choose).

If you've written some code but need help finishing it, want to get initial
feedback on it prior to finishing it, or want to share it and discuss prior
to finishing the implementation, you can open a *Draft* pull request and prepend
the title with the **\[WIP\]** tag (for Work In Progress). This will indicate
to reviewers that the code in the PR isn't in its final state and will change.
It also means that we will not merge the commit until it is finished. You or a
reviewer can remove the [WIP] tag when the code is ready to be fully reviewed for merging.

Before marking your Pull Request as "ready for review" make sure you have followed the
PR Checklist below. PRs that adhere to this list are more likely to get reviewed and
merged in a timely manner.

### Pull request checklist

When submitting a pull request and you feel it is ready for review,
please ensure that:

1. The code follows the code style of the project and successfully
   passes the CI tests. For convenience, you can execute `tox` locally,
   which will run these checks and report any issues.

   If your code fails the local style checks (specifically the black
   or Rust code formatting check) you can use `tox -eblack` and
   `cargo fmt` to automatically fix the code formatting.
2. The documentation has been updated accordingly. In particular, if a
   function or class has been modified during the PR, please update the
   *docstring* accordingly.

   If your pull request is adding a new class, function, or module that is
   intended to be user facing ensure that you've also added those to a
   documentation `autosummary` index to include it in the api documentation.
3. If you are of the opinion that the modifications you made warrant additional tests,
   feel free to include them
4. Ensure that if your change has an end user facing impact (new feature,
   deprecation, removal etc) that you have added a reno release note for that
   change and that the PR is tagged for the changelog.
5. All contributors have signed the CLA.
6. The PR has a concise and explanatory title (e.g. `Fixes Issue1234` is a bad title!).
7. If the PR addresses an open issue the PR description includes the `fixes #issue-number`
  syntax to link the PR to that issue (**you must use the exact phrasing in order for GitHub
  to automatically close the issue when the PR merges**)

### Code Review

Code review is done in the open and is open to anyone. While only maintainers have
access to merge commits, community feedback on pull requests is extremely valuable.
It is also a good mechanism to learn about the code base.

Response times may vary for your PR, it is not unusual to wait a few weeks for a maintainer
to review your work, due to other internal commitments. If you have been waiting over a week
for a review on your PR feel free to tag the relevant maintainer in a comment to politely remind
them to review your work.

Please be patient! Maintainers have a number of other priorities to focus on and so it may take
some time for your work to get reviewed and merged. PRs that are in a good shape (i.e. following the [Pull request checklist](#pull-request-checklist))
are easier for maintainers to review and more likely to get merged in a timely manner. Please also make
sure to always be kind and respectful in your interactions with maintainers and other contributors, you can read
[the Qiskit Code of Conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md).


## Contributor Licensing Agreement

Before you can submit any code, all contributors must sign a
contributor license agreement (CLA). By signing a CLA, you're attesting
that you are the author of the contribution, and that you're freely
contributing it under the terms of the Apache-2.0 license.

When you contribute to the Qiskit project with a new pull request,
a bot will evaluate whether you have signed the CLA. If required, the
bot will comment on the pull request, including a link to accept the
agreement. The [individual CLA](https://qiskit.org/license/qiskit-cla.pdf)
document is available for review as a PDF.

Note: If your contribution is part of your employment or your contribution
is the property of your employer, then you will more than likely need to sign a
[corporate CLA](https://qiskit.org/license/qiskit-corporate-cla.pdf) too and
email it to us at <qiskit@us.ibm.com>.

## Changelog generation

The changelog is automatically generated as part of the release process
automation. This works through a combination of the git log and the pull
request. When a release is tagged and pushed to github the release automation
bot looks at all commit messages from the git log for the release. It takes the
PR numbers from the git log (assuming a squash merge) and checks if that PR had
a `Changelog:` label on it. If there is a label it will add the git commit
message summary line from the git log for the release to the changelog.

If there are multiple `Changelog:` tags on a PR the git commit message summary
line from the git log will be used for each changelog category tagged.

The current categories for each label are as follows:

| PR Label               | Changelog Category |
| -----------------------|--------------------|
| Changelog: Deprecation | Deprecated         |
| Changelog: New Feature | Added              |
| Changelog: API Change  | Changed            |
| Changelog: Removal     | Removed            |
| Changelog: Bugfix      | Fixed              |

## Release notes

When making any end user facing changes in a contribution we have to make sure
we document that when we release a new version of qiskit. The expectation
is that if your code contribution has user facing changes that you will write
the release documentation for these changes. This documentation must explain
what was changed, why it was changed, and how users can either use or adapt
to the change. The idea behind release documentation is that when a naive
user with limited internal knowledge of the project is upgrading from the
previous release to the new one, they should be able to read the release notes,
understand if they need to update their program which uses qiskit, and how they
would go about doing that. It ideally should explain why they need to make
this change too, to provide the necessary context.

To make sure we don't forget a release note or if the details of user facing
changes over a release cycle we require that all user facing changes include
documentation at the same time as the code. To accomplish this we use the
[reno](https://docs.openstack.org/reno/latest/) tool which enables a git based
workflow for writing and compiling release notes.

### Adding a new release note

Making a new release note is quite straightforward. Ensure that you have reno
installed with:

    pip install -U reno

Once you have reno installed you can make a new release note by running in
your local repository checkout's root:

    reno new short-description-string

where short-description-string is a brief string (with no spaces) that describes
what's in the release note. This will become the prefix for the release note
file. Once that is run it will create a new yaml file in releasenotes/notes.
Then open that yaml file in a text editor and write the release note. The basic
structure of a release note is restructured text in yaml lists under category
keys. You add individual items under each category and they will be grouped
automatically by release when the release notes are compiled. A single file
can have as many entries in it as needed, but to avoid potential conflicts
you'll want to create a new file for each pull request that has user facing
changes. When you open the newly created file it will be a full template of
the different categories with a description of a category as a single entry
in each category. You'll want to delete all the sections you aren't using and
update the contents for those you are. For example, the end result should
look something like:

```yaml
features:
  - |
    Introduced a new feature foo, that adds support for doing something to
    :class:`.QuantumCircuit` objects. It can be used by using the foo function,
    for example::

      from qiskit import foo
      from qiskit import QuantumCircuit
      foo(QuantumCircuit())

  - |
    The :class:`.QuantumCircuit` class has a new method :meth:`~.QuantumCircuit.foo`. 
    This is the equivalent of calling the :func:`~qiskit.foo` to do something to your
    :class:`.QuantumCircuit`. This is the equivalent of running :func:`~qiskit.foo` 
    on your circuit, but provides the convenience of running it natively on
    an object. For example::

      from qiskit import QuantumCircuit

      circ = QuantumCircuit()
      circ.foo()

deprecations:
  - |
    The ``qiskit.bar`` module has been deprecated and will be removed in a
    future release. Its sole function, ``foobar()`` has been superseded by the
    :func:`~qiskit.foo` function which provides similar functionality but with
    more accurate results and better performance. You should update your
    :func:`~qiskit.bar.foobar` calls to :func:`~qiskit.foo`.
```

You can also look at other release notes for other examples.

For the ``features``, ``deprecations``, and ``upgrade`` sections there are a
list of subsections available which are used to provide more structure to the
release notes organization. If you're adding a feature, making an API change,
or deprecating an API you should pick the subsection that matches that note.
For example if you're adding a new feature to the transpiler, you should put
it under the ``upgrade_transpiler`` section.

Note that you can use sphinx [restructured text syntax](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).
In fact, you can use any restructured text feature in them (code sections, tables,
enumerated lists, bulleted list, etc) to express what is being changed as
needed. In general you want the release notes to include as much detail as
needed so that users will understand what has changed, why it changed, and how
they'll have to update their code.

After you've finished writing your release notes you'll want to add the note
file to your commit with `git add` and commit them to your PR branch to make
sure they're included with the code in your PR.

#### Linking to issues

If you need to link to an issue or other github artifact as part of the release
note this should be done using an inline link with the text being the issue
number. For example you would write a release note with a link to issue 12345
as:

```yaml
fixes:
  - |
    Fixes a race condition in the function ``foo()``. Refer to
    `#12345 <https://github.com/Qiskit/qiskit/issues/12345>` for more
    details.
```

#### Generating the release notes

After release notes have been added, you can use reno to see what the full output
of the release notes is. In general the output from reno that we'll get is a rst
(ReStructuredText) file that can be compiled by
[sphinx](https://www.sphinx-doc.org/en/master/). To generate the rst file you
use the ``reno report`` command. If you want to generate the full release
notes for all releases (since we started using reno during 0.9) you just run:

    reno report

but you can also use the ``--version`` argument to view a single release (after
it has been tagged:

    reno report --version 0.9.0

#### Building release notes locally

Building The release notes are part of the standard qiskit documentation
builds. To check what the rendered html output of the release notes will look
like for the current state of the repo you can run: `tox -edocs` which will
build all the documentation into `docs/_build/html` and the release notes in
particular will be located at `docs/_build/html/release_notes.html`

## Testing

Once you've made a code change, it is important to verify that your change
does not break any existing tests and that any new tests that you've added
also run successfully. Before you open a new pull request for your change,
you'll want to run the test suite locally.

The easiest way to run the test suite is to use
[**tox**](https://tox.readthedocs.io/en/latest/#). You can install tox
with pip: `pip install -U tox`. Tox provides several advantages, but the
biggest one is that it builds an isolated virtualenv for running tests. This
means it does not pollute your system python when running. Additionally, the
environment that tox sets up matches the CI environment more closely and it
runs the tests in parallel (resulting in much faster execution). To run tests
on all installed supported python versions and lint/style checks you can simply
run `tox`. Or if you just want to run the tests once run for a specific python
version: `tox -epy310` (or replace py310 with the python version you want to use,
py39 or py311).

If you just want to run a subset of tests you can pass a selection regex to
the test runner. For example, if you want to run all tests that have "dag" in
the test id you can run: `tox -epy310 -- dag`. You can pass arguments directly to
the test runner after the bare `--`. To see all the options on test selection
you can refer to the stestr manual:
https://stestr.readthedocs.io/en/stable/MANUAL.html#test-selection

If you want to run a single test module, test class, or individual test method
you can do this faster with the `-n`/`--no-discover` option. For example:

to run a module:
```
tox -epy310 -- -n test.python.compiler.test_transpiler
```
or to run the same module by path:

```
tox -epy310 -- -n test/python/compiler/test_transpiler.py
```
to run a class:

```
tox -epy310 -- -n test.python.compiler.test_transpiler.TestTranspile
```
to run a method:
```
tox -epy310 -- -n test.python.compiler.test_transpiler.TestTranspile.test_transpile_non_adjacent_layout
```

Alternatively there is a makefile provided to run tests, however this
does not perform any environment setup. It also doesn't run tests in
parallel and doesn't provide an option to easily modify the tests run.
For executing the tests with the makefile, a `make test` target is available.
The execution of the tests (both via the make target and during manual
invocation) takes into account the `LOG_LEVEL` environment variable. If
present, a `.log` file will be created on the test directory with the
output of the log calls, which will also be printed to stdout. You can
adjust the verbosity via the content of that variable, for example:

Linux and Mac:

``` {.bash}
$ cd out
out$ LOG_LEVEL="DEBUG" ARGS="-V" make test
```

Windows:

``` {.bash}
$ cd out
C:\..\out> set LOG_LEVEL="DEBUG"
C:\..\out> set ARGS="-V"
C:\..\out> make test
```

For executing a simple python test manually, we don\'t need to change
the directory to `out`, just run this command:

Linux and Mac:

``` {.bash}
$ LOG_LEVEL=INFO python -m unittest test/python/circuit/test_circuit_operations.py
```

Windows:

``` {.bash}
C:\..\> set LOG_LEVEL="INFO"
C:\..\> python -m unittest test/python/circuit/test_circuit_operations.py
```

##### STDOUT/STDERR and logging capture

When running tests in parallel using `stestr` either via tox, the Makefile
(`make test_ci`), or in CI we set the env variable
`QISKIT_TEST_CAPTURE_STREAMS` which will capture any text written to stdout,
stderr, and log messages and add them as attachments to the tests run so
output can be associated with the test case it originated from. However, if
you run tests with `stestr` outside of these mechanisms by default the streams
are not captured. To enable stream capture just set the
`QISKIT_TEST_CAPTURE_STREAMS` env variable to `1`. If this environment
variable is set outside of running with `stestr` the streams (STDOUT, STDERR,
and logging) will still be captured but **not** displayed in the test runners
output. If you are using the stdlib unittest runner a similar result can be
accomplished by using the
[`--buffer`](https://docs.python.org/3/library/unittest.html#command-line-options)
option (e.g. `python -m unittest discover --buffer ./test/python`).

##### Test Skip Options

How and which tests are executed is controlled by an environment
variable, `QISKIT_TESTS`:

Option | Description | Default
------ | ----------- | -------
`run_slow` | It runs tests tagged as *slow*. | `False`

It is possible to provide more than one option separated with commas.

Alternatively, the `make test_ci` target can be used instead of
`make test` in order to run in a setup that replicates the configuration
we used in our CI systems more closely.

### Snapshot Testing for Visualizations

If you are working on code that makes changes to any matplotlib visualizations
you will need to check that your changes don't break any snapshot tests, and add
new tests where necessary. You can do this as follows:

1. Make sure you have pushed your latest changes to your remote branch.
2. Go to link: `https://mybinder.org/v2/gh/<github_user>/<repo>/<branch>?urlpath=apps/test/ipynb/mpl_tester.ipynb`. For example, if your GitHub username is `username`, your forked repo has the same name the original, and your branch is `my_awesome_new_feature`, you should visit https://mybinder.org/v2/gh/username/qiskit/my_awesome_new_feature?urlpath=apps/test/ipynb/mpl_tester.ipynb.
This opens a Jupyter Notebook application running in the cloud that automatically runs
the snapshot tests (note this may take some time to finish loading).
3. Each test result provides a set of 3 images (left: reference image, middle: your test result, right: differences). In the list of tests the passed tests are collapsed and failed tests are expanded. If a test fails, you will see a situation like this:

   <img width="995" alt="Screenshot_2021-03-26_at_14 13 54" src="https://user-images.githubusercontent.com/23662430/112663508-d363e800-8e50-11eb-9478-6d665d0ff086.png">
4. Fix any broken tests. Working on code for one aspect of the visualizations
can sometimes result in minor changes elsewhere to spacing etc. In these cases
you just need to update the reference images as follows:
    - download the mismatched images (link at top of Jupyter Notebook output)
    - unzip the folder
    - copy and paste the new images into `qiskit/test/ipynb/mpl/references`,
  replacing the existing reference images
    - add, commit and push your changes, then restart the Jupyter Notebook app in your browser. The
  tests should now pass.
5. Add new snapshot tests covering your new features, extensions, or bugfixes.
    - add your new snapshot tests to `test/ipynb/mpl/test_circuit_matplotlib_drawer.py`
    , where you can also find existing tests to use as a guide.
    - commit and push your changes, restart the Jupyter Notebook app in your browser.
    As this is the first time you run your new tests there won't be any reference
    images to compare to. Instead you should see an option in the list of tests
    to download the new images, like so:

    <img width="1002" alt="Screenshot_2021-03-26_at_15 38 31" src="https://user-images.githubusercontent.com/23662430/112665215-b9c3a000-8e52-11eb-89e7-b18550718522.png">

    - download the new images, then copy and paste into `qiskit/test/ipynb/mpl/references`
    - add, commit and push your changes, restart the Jupyter Notebook app in your browser. The
    new tests should now pass.

Note: If you have run `test/ipynb/mpl_tester.ipynb` locally it is possible some file metadata has changed, **please do not commit and push changes to this file unless they were intentional**.


### Testing Rust components

Many Rust-accelerated functions are generally tested from Python space, but in cases
where new Rust-native APIs are being added, or there are Rust-specific internal details
to be tested, `#[test]` functions can be included inline. It's typically most
convenient to place these in a separate inline module that is only conditionally
compiled in, such as

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn my_first_test() {
        assert_eq!(2, 1 + 1);
    }
}
```

For more detailed guidance on how to add Rust testing you can refer to the Rust
documentation's [guide on writing tests](https://doc.rust-lang.org/book/ch11-01-writing-tests.html).

To run the Rust-space tests, do

```bash
cargo test --no-default-features
```

Our Rust-space components are configured such that setting the
``-no-default-features`` flag will compile the test runner, but not attempt to
build a linked CPython extension module, which would cause linker failures.

### Unsafe code and Miri

Any `unsafe` code added to the Rust logic should be exercised by Rust-space
tests, in addition to the more complete Python test suite.  In CI, we run the
Rust test suite under [Miri](https://github.com/rust-lang/miri) as an
undefined-behavior sanitizer.

Miri is currently only available on `nightly` Rust channels, so to run it
locally you will need to ensure you have that channel available, such as by
```bash
rustup install nightly --components miri
```

After this, you can run the Miri test suite with
```bash
MIRIFLAGS="<flags go here>" cargo +nightly miri test
```

For the current set of `MIRIFLAGS` used by Qiskit's CI, see the
[`miri.yml`](https://github.com/Qiskit/qiskit/blob/main/.github/workflows/miri.yml)
GitHub Action file.  This same file may also include patches to dependencies to
make them compatible with Miri, which you would need to temporarily apply as
well.

## Style and lint

Qiskit uses three tools for verify code formatting and lint checking. The
first tool is [black](https://github.com/psf/black) which is a code formatting
tool that will automatically update the code formatting to a consistent style.
The second tool is [pylint](https://www.pylint.org/) which is a code linter
which does a deeper analysis of the Python code to find both style issues and
potential bugs and other common issues in Python. The third tool is the linter
[ruff](https://github.com/charliermarsh/ruff), which has been recently
introduced into Qiskit on an experimental basis. Only a very small number
of rules are enabled.

You can check that your local modifications conform to the style rules by
running `tox -elint` which will run `black`, `ruff`, and `pylint` to check the
local code formatting and lint. If black returns a code formatting error you can
run `tox -eblack` to automatically update the code formatting to conform to the
style. However, if `ruff` or `pylint` return any error you will have to fix
these issues by manually updating your code.

Because `pylint` analysis can be slow, there is also a `tox -elint-incr` target,
which runs `black` and `ruff` just as `tox -elint` does, but only applies
`pylint` to files which have changed from the source github. On rare occasions
this will miss some issues that would have been caught by checking the complete
source tree, but makes up for this by being much faster (and those rare
oversights will still be caught by the CI after you open a pull request).

Because they are so fast, it is sometimes convenient to run the tools `black` and `ruff` separately
rather than via `tox`. If you have installed the development packages in your python environment via
`pip install -r requirements-dev.txt`, then `ruff` and `black` will be available and can be run from
the command line. See [`tox.ini`](tox.ini) for how `tox` invokes them.

### Rust style and lint

For formatting and lint checking Rust code, you'll need to use different tools than you would for Python. Qiskit uses [rustfmt](https://github.com/rust-lang/rustfmt) for
code formatting. You can simply run `cargo fmt` (if you installed Rust with the
default settings using `rustup`), and it will update the code formatting automatically to
conform to the style guidelines. This is very similar to running `tox -eblack` for Python code. For lint checking, Qiskit uses [clippy](https://github.com/rust-lang/rust-clippy) which can be invoked via `cargo clippy`. 

Rust lint and formatting checks are included in the the `tox -elint` command. For CI to pass you will need both checks to pass without any warnings or errors. Note that this command checks the code but won't apply any modifications, if you need to update formatting, you'll need to run `cargo fmt`.


## Building API docs locally

The API documentation is built with Sphinx.
We recommend that you use [**tox**](https://tox.readthedocs.io/en/latest) to orchestrate this.
Run a complete documentation build with
```
tox -e docs
```

The documentation output will be located at `docs/_build/html`.
Open the `index.html` file there in your browser to find the main page.

### Troubleshooting docs builds

When you build documentation, you might get errors that look like
```
ValueError: earliest-version set to unknown revision '1.0.0rc1'
```
If so, you need to fetch Qiskit's `git` tags and stable branches, in order to fully build the release notes.
To do this, run the command:
```
git fetch --tags upstream
```
where `upstream` is your name for the [git remote repository](https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes) that corresponds to https://github.com/Qiskit/qiskit (this repository).
You might need to re-run this command if Qiskit has issued a new release since the last time you built the documentation.

Sometimes, you might get errors about "names not existing" or "failed to import" during the docs build, even when the test suite passes.
This can mean that Sphinx's cache has become invalidated, but hasn't been successfully cleared.
Use the command:
```
tox -e docs-clean
```
to fully clean out all documentation build artefacts and partial builds, and see if the problem persists.


## Development cycle

The development cycle for qiskit is all handled in the open using
the project boards in Github for project management. We use milestones
in Github to track work for specific releases. The features or other changes
that we want to include in a release will be tagged and discussed in Github.
As we're preparing a new release we'll document what has changed since the
previous version in the release notes.

### Branches

* `main`:

The main branch is used for development of the next version of qiskit.
It will be updated frequently and should not be considered stable. The API
can and will change on main as we introduce and refine new features.

* `stable/*` branches:
Branches under `stable/*` are used to maintain released versions of qiskit.
It contains the version of the code corresponding to the latest release for
that minor version on pypi. For example, stable/0.8 contains the code for the
0.8.2 release on pypi. The API on these branches are stable and the only changes
merged to it are bugfixes.

### Release cycle

In the lead up to a release there are a few things to keep in mind. Prior to
the release date there is a feature, removal, and deprecation proposal freeze
date. This date in each release cycle is the last day where a new PR adding a
new feature, removing something, or adding a new deprecation can be proposed (in
a ready for review state) for potential inclusion in the release. If a new
PR is opened after this date it will not be considered for inclusion in that
release. Note, that meeting these deadlines does not guarantee inclusion in a
release: they are preconditions. You can refer to the milestone page for each
release to see these dates for each release (for example for 0.21.0 the page is:
https://github.com/Qiskit/qiskit/milestone/23).

After the proposal freeze a release review period will begin, during this time
release candidate PRs will be reviewed as we finalize the feature set and merge
the last PRs for the release. Following the review period a release candidate will be
tagged and published. This release candidate is pre-release that enables users and
developers to test the release ahead of time. When the pre-release is tagged the release
automation will publish the pre-release to PyPI (but only get installed on user request),
create the `stable/*` branch, and generate a pre-release changelog/release page. At
this point the `main` opens up for development of the next release. The `stable/*`
branches should only receive changes in the form of bug fixes at this point. If there
is a need additional release candidates can be published from `stable/*` and when the
release is ready a full release will be tagged and published from `stable/*`.

## Adding deprecation warnings
The qiskit code is part of Qiskit and, therefore, the [Qiskit Deprecation Policy](./DEPRECATION.md) fully applies here. Additionally, qiskit does not allow `DeprecationWarning`s in its testsuite. If you are deprecating code, you should add a test to use the new/non-deprecated method (most of the time based on the existing test of the deprecated method) and alter the existing test to check that the deprecated method still works as expected, [using `assertWarns`](https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertWarns). The `assertWarns` context will silence the deprecation warning while checking that it raises.

For example, if `Obj.method1` is being deprecated in favour of `Obj.method2`, the existing test (or tests) for `method1` might look like this:

```python
def test_method1(self):
   result = Obj.method1()
   self.assertEqual(result, <expected>)
```

Deprecating `method1` means that `Obj.method1()` now raises a deprecation warning and the test will not pass. The existing test should be updated and a new test added for `method2`:


```python
def test_method1_deprecated(self):
   with self.assertWarns(DeprecationWarning):
       result = Obj.method1()
   self.assertEqual(result, <expected>)

def test_method2(self):
   result = Obj.method2()
   self.assertEqual(result, <expected>)
```

`test_method1_deprecated` can be removed after `Obj.method1` is removed (following the [Qiskit Deprecation Policy](./DEPRECATION.md)).

## Using dependencies

We distinguish between "requirements" and "optional dependencies" in qiskit.
A requirement is a package that is absolutely necessary for core functionality in qiskit, such as Numpy or Scipy.
An optional dependency is a package that is used for specialized functionality, which might not be needed by all users.
If a new feature has a new dependency, it is almost certainly optional.

### Adding a requirement

Any new requirement must have broad system support; it needs to be supported on all the Python versions and operating systems that qiskit supports.
It also cannot impose many version restrictions on other packages.
Users often install qiskit into virtual environments with many different packages in, and we need to ensure that neither we, nor any of our requirements, conflict with their other packages.
When adding a new requirement, you must add it to [`requirements.txt`](requirements.txt) with as loose a constraint on the allowed versions as possible.

### Adding an optional dependency

New features can also use optional dependencies, which might be used only in very limited parts of qiskit.
These are not required to use the rest of the package, and so should not be added to `requirements.txt`.
Instead, if several optional dependencies are grouped together to provide one feature, you can consider adding an "extra" to the package metadata, such as the `visualization` extra that installs Matplotlib and Seaborn (amongst others).
To do this, modify the [`setup.py`](setup.py) file, adding another entry in the `extras_require` keyword argument to `setup()` at the bottom of the file.
You do not need to be quite as accepting of all versions here, but it is still a good idea to be as permissive as you possibly can be.
You should also add a new "tester" to [`qiskit.utils.optionals`](qiskit/utils/optionals.py), for use in the next section.

### Checking for optionals

You cannot `import` an optional dependency at the top of a file, because if it is not installed, it will raise an error and qiskit will be unusable.
We also largely want to avoid importing packages until they are actually used; if we import a lot of packages during `import qiskit`, it becomes sluggish for the user if they have a large environment.
Instead, you should use [one of the "lazy testers" for optional dependencies](https://docs.quantum.ibm.com/api/qiskit/utils#optional-dependency-checkers), and import your optional dependency inside the function or class that uses it, as in the examples within that link.
Very lightweight _requirements_ can be imported at the tops of files, but even this should be limited; it's always ok to `import numpy`, but Scipy modules are relatively heavy, so only import them within functions that use them.


## Dealing with the git blame ignore list

In the qiskit repository we maintain a list of commits for git blame
to ignore. This is mostly commits that are code style changes that don't
change the functionality but just change the code formatting (for example,
when we migrated to use black for code formatting). This file,
`.git-blame-ignore-revs` just contains a list of commit SHA1s you can tell git
to ignore when using the `git blame` command. This can be done one time
with something like

```
git blame --ignore-revs-file .git-blame-ignore-revs qiskit/version.py

```

from the root of the repository. If you'd like to enable this by default you
can update your local repository's configuration with:

```
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

which will update your local repositories configuration to use the ignore list
by default.
