共有三個.py的檔案，使用方式如下。
1. DataPreprocessing.py
跑全部程式(按F5)之後，會在相對路徑生成train_modified_v5.csv和test_modified_v5.csv，分別代表training data和testing data經過處理修正後的資料集。
而細部程式會以#%%做區隔，其中的對應的comment會表示該模塊的運作，主要有資料視覺化、資料轉換、缺失值處理、資料工程(Data Engineering)、統計檢定和產生圖表(Plotting)的過程。

2. decisiontree.py
參數:
 (1)檔案名稱:23行(訓練資料檔案名稱)
 (2)欲選特徵:24行(Survived+特徵)、27行(特徵)
 (3)產生csv檔名字:265行
 (4)訓練資料檔案名稱:279行
 (5)測試資料檔案名稱:282行
 (6)最低準確度:277行
 (7)最小信息增益:289行

3. naivebayes.py
執行程式後會跑出預測結果(.csv)。