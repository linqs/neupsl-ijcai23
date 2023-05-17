# ===============================================
# NeuPSL
# ===============================================

python3 parse-results.py

# ===============================================
# Baselines
# ===============================================

# Citation
# ---------------------------

cd ../citation/other-methods/deepstochlog/scripts
python3 parse-results.py
cd ../../gnn/scripts
python3 parse-results.py
cd ../../psl/scripts
python3 parse-results.py
cd ../../../../scripts

# MNIST Addition
# ---------------------------

cd ../mnist-addition/other-methods/cnn/scripts
python3 parse-results.py
cd ../../deepproblog/scripts
python3 parse-results.py
cd ../../ltn/scripts
python3 parse-results.py
cd ../../../../scripts

# VSPC
# ---------------------------

cd ../vspc/other-methods/cnn-digit/scripts
python3 parse-results.py
cd ../../cnn-visual/scripts
python3 parse-results.py
cd ../../../../scripts