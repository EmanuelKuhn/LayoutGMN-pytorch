# LayoutGMN: Neural Graph Matching for Structural Layout Similarity
This repo provides the source code for our **[CVPR 2021 paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Patil_LayoutGMN_Neural_Graph_Matching_for_Structural_Layout_Similarity_CVPR_2021_paper.pdf)**.

Overall, the repo consists of three parts:

1) Preparing Layout Graphs
2) Leveraging Graph Matching Networks (GMN)
3) Training GMN on the layout graphs

# Preparing Layout Graphs
We used two kinds of layout data in our work: Floorplans and UI designs.
Our code for layout graph data preparation is, in parts, borrowed from the work of **[Dipu et al](https://github.com/dips4717/gcn-cnn)**.

In case you are looking for apn_dict (anchor-pos-neg) on floorplan data, here is the **[link](https://drive.google.com/file/d/1f-4Ll7HOKsZ2gJQgu9QZD7_MNCO3OOvV/view?usp=sharing)** to download it.

A few other pickle files that are needed for training on RPLAN data include: **[FP_box_info.pkl](https://drive.google.com/file/d/1rRw6XBsSZv9nnHSjj6YTvS-fqvAm_3WT/view?usp=sharing)**, **[FP_box_info_list.pkl](https://drive.google.com/file/d/10Q4EjqqE3kFjP0BTdnE2s0wZ1xycKndG/view?usp=sharing)** and **[FP_data.p](https://drive.google.com/file/d/15U4B59mz54qWIaQXDFV-xXi1Vchn8y_D/view?usp=sharing)**

# Preprocessing RPLAN floorplans

See [gcn-cnn preprocessing](https://github.com/EmanuelKuhn/gcn-cnn/tree/master/preprocessing) for how to prepare your own training data.

# Graph Matching Network-pytorch
The main machinery in our work is a Graph Matching Network that operates on the obtained layout graphs. 
We provide our own PyTorch implementation of Graph Matching Networks built on top of the Tensorflow **[Colab implementation](https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/graph_matching_networks/graph_matching_networks.ipynb)** by DeepMind Research.

## Requirements
All the required modules are present in the `requirements.txt` file. 
Install all the requirements via

`pip install -r requirements.txt`

You should mainly care about these:
Pytorch >=1.6, CUDA >= 9.1, networkx >= 2.3, torch-sparse==0.6.7 (pip install torch-sparse), torch-cluster==1.4.5 (pip install torch-cluster), torch-geometric==1.3.2 (pip install torch-geometric)


## Citation
If you find our work useful in your research, consider citing:

```
@InProceedings{Patil_2021_CVPR,
    author    = {Patil, Akshay Gadi and Li, Manyi and Fisher, Matthew and Savva, Manolis and Zhang, Hao},
    title     = {LayoutGMN: Neural Graph Matching for Structural Layout Similarity},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {11048-11057}
}
```
