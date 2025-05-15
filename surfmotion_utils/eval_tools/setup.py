from setuptools import find_packages, setup

package_name = 'eval_tools'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + "/output", ["output/.gitkeep"]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='carl',
    maintainer_email='lovehult98@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joint_states_logger = eval_tools.joint_states_logger:main',
            'eef_logger = eval_tools.eef_logger:main'
        ],
    },
)
