import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

     name='pvtseg',  

     version='1.0',
     

     author="Poquillon Titouan",

     author_email="titouan.poquillon@gmail.com",

     description="A segmentation pipeline of the pulmonary vascular tree",

     long_description=long_description,

     long_description_content_type="text/markdown",

     url="https://gitlab.in2p3.fr/odyssee.merveille/pvt-segmentation",

     packages=setuptools.find_packages(),

     
     install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
          'diplib',
          'shap',
          'pathlib',
          'sklearn'
     ],

     classifiers=[

         "Programming Language :: Python :: 3.7",

         "License :: OSI Approved :: Inria License",

         "Operating System :: All",

     ],

 )