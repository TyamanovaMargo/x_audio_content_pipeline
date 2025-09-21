# Запуск с запасным аккаунтом
% python test.py --input sample_usernames.csv --output results.csv \ 
  --login margati@ac.sce.ac.il --password 15092025 \  
  --backup-login testbuba23@gmail.com --backup-password 15092025bubatest1 \ 
  --max-concurrent 2

python test.py --input sample_usernames.csv --output results.csv --login margati@ac.sce.ac.il --password 15092025 --backup-login testbuba23@gmail.com --backup-password 15092025bubatest1 --no-headless




# Продолжение после остановки
python test.py --input usernames.csv --output results.csv --resume

# Через переменные окружения для безопасности
export X_LOGIN="main_account"
export X_PASSWORD="main_pass" 
export X_BACKUP_LOGIN="backup_account"
export X_BACKUP_PASSWORD="backup_pass"
python test.py --input usernames.csv --output results.csv --max-concurrent 2






# To use your old environment (Python 3.13)
conda deactivate
source .venv/bin/activate

# To use the NeMo environment (Python 3.12)
deactivate  # if in venv
conda activate nemo_env
