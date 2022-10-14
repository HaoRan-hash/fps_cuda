# fps_cuda

#### Requirements

Pytorch

#### Install

```python
python setup.py install
# or
pip install .
```

#### Introduction

With the popularity of deep learning in recent years, many point cloud algorithms also use deep learning. **In point cloud algorithm, Farthest Point Sampling (FPS) is very commom.** The time complexity of the FPS algorithm is positively related to the size of the point cloud.

Many current implementations are based on python, which are very slow. I implement the FPS algorithm using **cuda** and the performance is about **30x faster** than the python version. After testing, the result is correct.

If you are troubled by the time, try it, there is usage example in `test` folder (you should have [Stanford Large-Scale Indoor Spaces 3D](https://cvgl.stanford.edu/resources.html) dataset first).
