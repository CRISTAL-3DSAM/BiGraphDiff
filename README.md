
# InterFormer

Code for the paper "Bipartite Graph Diffusion Model for Human Interaction Generation" ([paper](https://openaccess.thecvf.com/content/WACV2024/papers/Chopin_Bipartite_Graph_Diffusion_Model_for_Human_Interaction_Generation_WACV_2024_paper.pdf)))


**dependencies:**
- python 3.8
- torch 1.8.1 with cuda enabled
- numpy
- scipy
- imageio
- matplotlib
- clip
- sklearn
- tensorflow-gpu 2.10.0 (FVD/multimodality calculation only)
- tensorflow_gan 2.1.0 (FVD/multimodality calculation only)

To generate sequence for NTU_26:
in BiGraphDiff_NTU_26\BiGraphDiff run

```
	python test_a2m_transf_NTU.py -load_weights save -batch_size 2
```
The generated sequence are save in BiGraphDiff_NTU_26\BiGraphDiff\data_convert
	
To generate sequence for DuetDance:
in BiGraphDiff_DuetDance\BiGraphDiff run

```
	python test_a2m_transf_DD.py -load_weights save -batch_size 128
```

The generated sequence are save in BiGraphDiff_DuetDance\BiGraphDiff\data_convert

This was tested on a NVIDIA A100 80Go GPU. Adapt the batch_size to your own configuration.
Rising the number of epochs for NTU_26 enable a faster generation but lower the quality of the results. This is because we generate sequence from the maximum length form the batch which can not be fit for certain classes

To get the classification accuracy on NTU_26: 
in BiGraphDiff_NTU_26\Classifier run

```
	python test.py -load_weights save
```


 **Citation:**
```
@inproceedings{chopin2024bipartite,
  title={Bipartite graph diffusion model for human interaction generation},
  author={Chopin, Baptiste and Tang, Hao and Daoudi, Mohamed},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5333--5342},
  year={2024}
}
```

**Acknowledgements**

This project has received financial support from the CNRS through the 80—Prime program, from the French State, managed by the National Agency for Research (ANR) under the Investments for the future program with reference ANR-16-IDEX-0004 ULNE and by the EU H2020 project AI4Media under Grant 951911. 
