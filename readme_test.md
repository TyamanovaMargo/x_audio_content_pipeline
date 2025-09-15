# Запуск с запасным аккаунтом
python test.py --input usernames.csv --output results.csv \
  --login main_account --password main_pass \
  --backup-login backup_account --backup-password backup_pass \
  --max-concurrent 2

# Продолжение после остановки
python test.py --input usernames.csv --output results.csv --resume

# Через переменные окружения для безопасности
export X_LOGIN="main_account"
export X_PASSWORD="main_pass" 
export X_BACKUP_LOGIN="backup_account"
export X_BACKUP_PASSWORD="backup_pass"
python test.py --input usernames.csv --output results.csv --max-concurrent 2
