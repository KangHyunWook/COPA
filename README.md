# COPA


<p align="center">
  <img width="800" src="COPA.png">
</p>

### Data Download and set up environments

 - [SEED](https://bcmi.sjtu.edu.cn/home/seed/seed.html)
 - [SEED-IV](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html#)


#Running the code
```
python main.py --model_name Conformer --dataset_dir $FEIS --num_electrodes 14 --n_subjects 21 --exper-setting indep --save_file_name indep_conformer_results.csv
```


### Citation

If this code is helpful for your research, please cite us at:

```
@article{Kang2026COPA,
  title={Incongruity-aware shared-private parallel fusion with selective feature passing for EEG-based emotion recognition},
  author={Kang, Hyunwook and Lee, Young-Eun and Lee, Minji},
  journal={Biomedical Signal Processing and Control},
  year={2026}
}
```

### Contact

For any questions, please email at [hyunwook.kang@catholic.ac.kr](mailto:hyunwook.kang@catholic.ac.kr)
