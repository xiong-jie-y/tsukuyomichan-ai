import setuptools

__version__ = '0.0.1dev1'

setuptools.setup(
    name='tsukuyomichan_ai',
    version=__version__,
    packages=setuptools.find_packages(),
    py_modules=["generate_talking_video", "agent_display"]
)