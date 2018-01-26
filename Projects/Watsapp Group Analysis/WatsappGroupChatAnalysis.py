import pandas as pd

df=pd.read_table(r'D:\Projects\Python\All-Machine-Learning\Projects\Watsapp Group Analysis\WhatsApp Chat.txt',error_bad_lines=False)

print(df.head())