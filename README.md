Replication of the following paper using Pytorch:
![image](https://github.com/radoslaw626/ViT_paper_replication/assets/50368353/a7c4a5da-dafd-4dc0-a7e2-5d910c3c627f)

The notebook contains an analysis of each equation that the paper covers. Each equation is represented as a block, where I test
the shapes of inputs and outputs, and how the data changes.
![image](https://github.com/radoslaw626/ViT_paper_replication/assets/50368353/d8394ada-ff55-402d-8868-c99334f23686)
By the end of the notebook, ViT is completed in variant ViT-B(16). Training this model would take more than 30 days on my machine,
so I used a pre-trained ViT-B model from Torchvision. I compared the model's layers to what I have achieved by following paper.
The project uses scripts and data that I have prepared in my other projects (TinyVGG).
