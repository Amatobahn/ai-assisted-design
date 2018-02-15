from distutils.core import setup

setup(
    name='PyWebDevelopment',
    version='0.1dev',

    description='Explorations using Python for the purposes of quick prototyping webviews.',

    # The project's main homepage.
    url='http://www.iamgregamato.com',

    # Author details.
    author='Greg Amato',
    author_email='amatobahn@gmail.com',

    # License
    license='Proprietary License',

    # Classifiers
    classifiers=[
        # Project Stage:
        'Development Status :: 3 - Alpha',

        # Intended for:
        'Intended Audience :: Learning',
        'Topic :: Software Development :: Education',

        # License:
        'License :: Proprietary License',

        # Supported Python versions:
        'Programming Language :: Python :: 3.5.4',
    ],

    # Keywords
    keywords='education neural net ai opencv',

    # Required dependencies. Will be installed by pip
    # when the project is installed.
    install_requires=['numpy', 'pandas', 'opencv-python', 'pypiwin32',
                      'tensorflow', 'tflearn', 'yattag'],
)
