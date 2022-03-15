from setuptools import setup, find_packages


def readme():
    with open('lane_line_recognition/README.md') as f:
        return f.read()


def requires():
    with open('lane_line_recognition/requirements.txt') as f:
        return f.read().splitlines()


setup(
    name='lane_line_recognition',
    version='1.0',
    packages=find_packages(exclude=['*.env', 'old_code', '.idea', 'lane_line']),
    install_requires=requires(),
    url='',
    license='',
    author='Karim Safiullin',
    author_email='st.herbinar@icloud.com',
    python_requires='>=3',
    zip_safe=True,
    description='Install line lane recognition',
    long_description=readme(),
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Topic :: Software Development',
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3.7',
        'Environment :: Console',
    ],
)
