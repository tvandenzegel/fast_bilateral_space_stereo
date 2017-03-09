# Fast Bilateral-Space Stereo
A naïve implementation of the Fast Bilateral-Space Stereo paper by Jonathan T. Barron. [[1]](http://jonbarron.info/BarronCVPR2015.pdf)[[2]](http://jonbarron.info/BarronCVPR2015_supp.pdf)[[3]](http://jonbarron.info/)   
The goal of this project was to get a real understanding and practical experience of this method.  
Only the simplified bilateral grid method without the multiscale optimization is implemented.  
The algorithm needs a stereo pair as input and will generate a disparity map.

![Overview screenshot](data/middlebury_summary.jpg?raw=true "Overview screenshot")

## Dependencies
The following dependencies are needed.  
The version numbers are the ones used during development.  
- OpenCV 3.2.0
  - (optional) opencv_contrib (if you want to use the domain transform filter to smoothen the disparity map)
- Eigen 3.3.2
- Ceres Solver 1.12.0
  - Glog 0.3.3
  - GFlags 2.2.0

The code was developed on a Windows machine with Visual Studio 2015.  

## References
[[1]](http://jonbarron.info/BarronCVPR2015.pdf) Barron, Jonathan T., et al. "Fast bilateral-space stereo for synthetic defocus." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.* 2015.

[[2]](http://jonbarron.info/BarronCVPR2015_supp.pdf) Barron, Jonathan T., et al. "Fast bilateral-space stereo for synthetic defocus—Supplemental material." *Proc. IEEE Conf. Comput. Vis. Pattern Recognit.(CVPR).* 2015.