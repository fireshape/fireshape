from setuptools import setup
#try:
#    import firedrake # noqa
#    import firedrake.adjoint # noqa
#    import ROL # noqa
#except ImportError:
#    raise Exception("Firedrake needs to be installed and activated."
#                    "Please visit firedrakeproject.org ."
#                    "ROL must have been installed, too")
setup(
    name='fireshape',
    version='0.0.1',
    author='Alberto Paganini, Florian Wechsung',
    author_email='admp1@leicester.ac.uk',
    description='A library for shape optimization based on firedrake',
    long_description='',
    packages=['fireshape', 'fireshape.zoo'],
    zip_safe=False,
    install_requires=["scipy"]
)
