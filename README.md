# The landscape of artificial intelligence in carbon ion radiotherapy: a scoping review
Repository supporting the article submitted to La Radiologia Medica

If you use this codebase for your research, please cite our paper if available; otherwise, please cite this repository:
```bibtex
TBA
```
### **Repository structure**
#### **Overview:**

* LLM for screening articles (using the title + abstract)
* LLM for complete critical review (using the complete text, excluding the references section)
* Visualizations
* Jupyter notebooks with usage examples

```bibtex
Note: We used DeepSeek-R1:32B for all our analysis. Ollama was used to run the LLM locally.
```

#### **Contents:**
```
llm_review
├── database                           # database directory
│    └── articles_to_screen.csv        #----Rayyan exported article title and abstracts for screening
│    └── articles_to_review.xlsx       #----Excel file with index to the selected articles
│    └── selected_articles             #----directory containing the PDFs of the 19 selected articles
├── notebooks                          # directory containing Jupyter notebook, results, and figures
│    └── llm_reviewer.ipynb            #----Jupyter notebook with example usage of LLM-assisted screening, and complete review
│    └── analysis.ipynb                #----Jupyter notebook for visualization and agreement analysis
│    └── results                       #---- directory containing the results of this scoping review
│    └── figures                       #---- directory containing trend and wordcloud plots
├── README.md

```

