# Fast Bilateral Space Stereo
## Introduction
A naïve implementation of the Fast Bilateral-Space Stereo paper by Jonathan T. Barron. [(link to paper)](http://jonbarron.info/BarronCVPR2015.pdf) [(link to Jon Barron personal page)](http://jonbarron.info/).
The goal of this project was to get practical experience and understanding of the paper.
Only the simplified bilateral grid without the multiscale optimization is implemented.
The algorithm needs a stereo pair as input and the output is a disparity map.


![Alt text](data/result_3d.jpg?raw=true "Title")\go

## Dependencies

- OpenCV
- Eigen
- Ceres Solver
  - Glog
  - GFlags
  
## References
[[1]](http://jonbarron.info/BarronCVPR2015.pdf) Barron, Jonathan T., et al. "Fast bilateral-space stereo for synthetic defocus." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.* 2015.

[[2]](http://jonbarron.info/BarronCVPR2015_supp.pdf) Barron, Jonathan T., et al. "Fast bilateral-space stereo for synthetic defocus—Supplemental material." *Proc. IEEE Conf. Comput. Vis. Pattern Recognit.(CVPR).* 2015.