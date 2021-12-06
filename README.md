# McGiraffe: Mask Based Composition Training For Figure-Ground Disentanglement in Generative Neural Feature Fields

This code is based on the original GIRAFFE implementation made by Niemeyer and Geiger [Here](https://github.com/autonomousvision/giraffe)

## Preparation of the DB, models and enviroment

To allow the repository to succesfully run the databases and models must be added. Also, to avoid problems regarding libraries we give one of our enviroments to use. This are the lines that must be runed:
### BCV002
```
ln -s "/media/bcv003/user_home0/jmabril/Proyecto_AML/out/"
cd data 
ln -s "/media/bcv003/user_home0/jmabril/Proyecto_AML/data/comprehensive_cars/"
cd ..
conda activate "/media/user_home0/jmabril/anaconda3/envs/giraffe/"
```

### BCV003
```
ln -s "/media/user_home0/jmabril/Proyecto_AML/out/"
cd data 
ln -s "/media/user_home0/jmabril/Proyecto_AML/data/comprehensive_cars/"
cd ..
conda activate "/media/bcv002/user_home0/jmabril/anaconda3/envs/giraffe/"
```

### Test

To test the method the following command must be runed. This command creates a new directory named test, here you are going to find one image that contain 4 different images constructed with both composition methods, the segmented car and the sigma map. Also, you will find a new file called 'rotation_object' you are going to find rotation images and videos of cars. Finally you will find in the same folder an image containing 256 of the 20.000 used to compute the FID and you will get in the console the calculated FID.
The code to run this mode is: 

>python main.py --mode test

### Demo

To perform the demo the following comand must be executed.  This command creates a new directory named demo, here you will find one image that contain a car constructed with both composition methods, the segmented car and the sigma map. Also, you will find a new file called 'rotation_object' where you will find a video of the same car rotating.The code to run this mode is: 

>python main.py --mode demo
...

For this, one of the test images can be used:

    'lung_003.nii.gz', 'lung_009.nii.gz', 'lung_055.nii.gz', 'lung_059.nii.gz', 'lung_079.nii.gz', 'lung_081.nii.gz', 'lung_093.nii.gz'.
