
# BiGraphDiff

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



**Data and weight**

We do not provide the data for DuetDance and NTU as it is not public. 
You can obtain the NTU dataset at ([NTU](https://rose1.ntu.edu.sg/dataset/actionRecognition/))). You need to download the skeleton data for the 26 classes labeled "Mutual Actions / Two Person Interactions". **Preprocessing code coming soon**.
To obtain the format used in the code you need to extract the euclidean coordinates for each skeletons and remove the hand, feet and head joints and then normalize with normalize.m. The files should be named skeleton_A.mat and skeleton_B.mat

You can ask the authors of DuetDance for the raw data and then use format_files.py followed by normalize.m to obtain the same data as the one we used. Then put this data into BiGraphDiff_DuetDance\BiGraphDiff\data\Skeletons

The weight can be found here: [Google Drive](https://drive.google.com/drive/folders/1ekHkzKg69yykWjAQ9RqAglm7ePWt1qjw?usp=sharing)




**Testing**

To generate sequence for NTU_26:
```
in BiGraphDiff_NTU_26\BiGraphDiff run
	python test_a2m_transf_NTU.py -load_weights save -batch_size 2
```

The generated sequence are save in BiGraphDiff_NTU_26\BiGraphDiff\data_convert
	
To generate sequence for DuetDance:
```
in BiGraphDiff_DuetDance\BiGraphDiff run
	python test_a2m_transf_DD.py -load_weights save -batch_size 128
```
 
The generated sequence are save in BiGraphDiff_DuetDance\BiGraphDiff\data_convert

This was tested on a NVIDIA A100 80Go GPU. Adapt the batch_size to your own configuration.
Rising the number of epochs for NTU_26 enable a faster generation but lower the quality of the results. This is because we generate sequence from the maximum length form the batch which can not be fit for certain classes

To get the classification accuracy on NTU_26: 
```
in BiGraphDiff_NTU_26\Classifier run
	python test.py -load_weights save
```

To get the classification accuracy on DuetDance: 
```
in BiGraphDiff_DuetDance\Classifier run
	python test_DD.py -load_weights save
```
	
By default we put our generated results. You need to run the generation code to overwrite them.

To get the FVD and Multimodality evaluation on NTU_26:
```
in BiGraphDiff_NTU_26\Classifier run
	python compute_FVD.py
```
	
To get the FVD and Multimodality evaluation on DuetDance:
```
in BiGraphDiff_DuetDance\Classifier run
	python compute_FVD.py
```
	
To generate  visual in .gif format for NTU_26:
```
in BiGraphDiff_NTU_26 run
	python Generate_Bigraph_visuals.py
```

The results will be saved in BiGraphDiff_NTU_26/Visuals

To generate  visual in .gif format for DuetDance:
```
in BiGraphDiff_DuetDance run
	python Generate_Bigraph_DD.py
```

The results will be saved in BiGraphDiff_DuetDance/Visuals

By default we generate the result of the entire test set. It can take a long time especially for NTU_26.
You can reduce the numbers by BiGraphDiff_DuetDance\Classifier\data or BiGraphDiff_DuetDance\Classifier\data create a copy of Test_generated.txt and remove some lines.
Then you need to replace Test_generated.txt by the name of your new file in Generate_Bigraph_DD.py line 65 or Generate_Bigraph_visuals.py line 52



We do not provide the data for DuetDance as it is not public. You can ask the authors for the raw data and then use format_files.py followed by normalize.m to obtain the same data as the one we used. Then put this data into BiGraphDiff_DuetDance\BiGraphDiff\data\Skeletons

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

This project has received financial support from the CNRS through the 80—Prime program, from the French State, managed by the National Agency for Research (ANR) under the Investments for the future program with reference ANR-16-IDEX-0004 ULNE.
