# Path-Tracer
A simple rendering program based on MIS path tracing

## How to Run the program
The environment on my PC is based on VS2015. If you want to reproduce the result, you can either compile 
```ques_h2.cpp```
on VS2015 or higher version then press CTRL + F5 or compile it with a GCC compiler

To set the max running time, change the value of 
```float maxTime```
And the default value will be 60 seconds

## About the Math Utilities
Some of the math utilities like Vec3f, Mat4f and GeometryList are taken from open source code of smallVCM.
*http://www.smallvcm.com*
Corresponding referrences have been noted in the code

## About the Scene
The scene and the camera are totally built by myself.
**Models include:** 
a traditional Cornell Box
a diffusal white cude
a glass cude
several glass balls
several mirror balss (IOR = 1.6)

## About the BSDF
All the BSDF implementations are guided by 
*Physically Based Rendering From Theory to Implementation 3rd Editon - Matt Pharr, Wenzel Jakob, Greg Humphreys*
You can find most of the formulations of sampling and evaluation in Chap7 and Chap8

## About the Renderer
The renderer employs ray casting algorithm where rays are generated from the view point instead of from the light source
**simple illustration of the render pipeline:**
generate ray -> 
ray intersect with material -> 
meterial BSDF -> 
add color ->
get new ray -> 
check whether path length < 10 (max length)

## About the Result picture
I ran the renderer on a PC with Intel Core i7 - 7770k for 1h and got a final example BMP