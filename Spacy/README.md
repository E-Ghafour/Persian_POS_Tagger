For training you should first create `.spacy` files for training and testing by using the available codes in notebooks and then, run these commands:

```bash
python -m spacy init fill-config base_config.cfg confic.cfg
python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./test.spacy 
```

If you want to use gpu instead of cpu, you can utilize dedicated gpu files. 
