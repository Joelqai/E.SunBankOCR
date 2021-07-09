import os
import setuptools

setuptools.setup(
    name = 'HandWritingCategory',
    version='1.1.0',
    description='multiple classification for hand writing',
    long_description=open(
        os.path.join(
            os.path.dirname(__file__),
            'README.md'
        )
    ).read(),
    author='huangtingshieh',
    author_email='huangtingshieh@gmail.com',
    packages=setuptools.find_packages(),
    license='NO',
)