## HELPFUL , INORDER TO EXPORT YOUR PROJECT AS A PACKAGE , SO THAT ANYONE ELSE CAN USE IT 

from setuptools import find_packages,setup
from typing import List
HYPHEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:

    #this fxn will retun list of requirements
   
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n" , "")for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements
setup(
    name="mlproject",
    version='0.0.1',
    author='Siya Pari',
    author_email='siyachandel56@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)