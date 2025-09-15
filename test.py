#!/usr/bin/env python3

import asyncio
import csv
import os
import sys
import argparse
from datetime import datetime
from typing import Optional, Dict
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
import random
import time

# Человекоподобное поведение: случайные задержки, скроллы, движения мыши
async def human_like_behavior(page: Page):
    """Имитирует человеческое поведение на странице"""
    await page.mouse.move(random.randint(100, 400), random.randint(100, 400))
    await asyncio.sleep(random.uniform(0.5, 1.5))
    await page.mouse.move(random.randint(100, 800), random.randint(100, 800))
    await asyncio.sleep(random.uniform(0.5, 1.5))
    await page.evaluate('window.scrollBy(0, Math.floor(Math.random() * 100));')
    await asyncio.sleep(random.uniform(0.3, 1.0))

async def login_x(context: BrowserContext, headless: bool, login: str, password: str) -> BrowserContext:
    """Улучшенная авторизация в X.com"""
    page = await context.new_page()
    
    try:
        print("🔐 Переход на страницу входа...")
        await page.goto('https://x.com/i/flow/login', timeout=20000)
        await asyncio.sleep(random.uniform(3.0, 5.0))

        # Попробуем разные селекторы для поля username
        username_selectors = [
            'input[name="text"]',
            'input[autocomplete="username"]',
            'input[data-testid="ocfEnterTextTextInput"]',
            'input[type="text"]'
        ]
        
        username_input = None
        for selector in username_selectors:
            try:
                username_input = await page.wait_for_selector(selector, timeout=5000)
                print(f"✅ Найдено поле username: {selector}")
                break
            except:
                continue
        
        if not username_input:
            print("❌ Не найдено поле для ввода имени пользователя")
            if not headless:
                print("🔍 Браузер остается открытым для ручного входа")
                input("Выполните вход вручную и нажмите Enter для продолжения...")
                await page.close()
                return context
            else:
                print("⚠️ Продолжаем без авторизации")
                await page.close()
                return context
            
        await username_input.fill(login)
        await human_like_behavior(page)
        
        # Поиск кнопки "Next"
        next_selectors = [
            'div[role="button"]:has-text("Next")',
            'div[role="button"]:has-text("Далее")',
            'button:has-text("Next")',
            '[data-testid="LoginForm_Login_Button"]'
        ]
        
        for selector in next_selectors:
            try:
                next_button = await page.wait_for_selector(selector, timeout=3000)
                await next_button.click()
                print("✅ Нажата кнопка Next")
                break
            except:
                continue
        
        await asyncio.sleep(random.uniform(2.0, 4.0))

        # Поиск поля пароля с разными селекторами
        password_selectors = [
            'input[name="password"]',
            'input[type="password"]',
            'input[autocomplete="current-password"]',
            'input[data-testid="ocfEnterTextTextInput"]'
        ]
        
        password_input = None
        for selector in password_selectors:
            try:
                password_input = await page.wait_for_selector(selector, timeout=8000)
                print(f"✅ Найдено поле пароля: {selector}")
                break
            except:
                continue
        
        if not password_input:
            print("❌ Не найдено поле для ввода пароля")
            if not headless:
                print("🔍 Браузер остается открытым для ручного входа")
                input("Выполните вход вручную и нажмите Enter для продолжения...")
                await page.close()
                return context
            else:
                print("⚠️ Продолжаем без авторизации")
                await page.close()
                return context
            
        await password_input.fill(password)
        await human_like_behavior(page)
        
        # Поиск кнопки входа
        login_selectors = [
            'div[role="button"]:has-text("Log in")',
            'div[role="button"]:has-text("Войти")',
            'button[type="submit"]',
            '[data-testid="LoginForm_Login_Button"]'
        ]
        
        for selector in login_selectors:
            try:
                login_button = await page.wait_for_selector(selector, timeout=5000)
                await login_button.click()
                print("✅ Нажата кнопка входа")
                break
            except:
                continue

        # Ожидание завершения входа
        await asyncio.sleep(8)

        # Проверка 2FA
        try:
            challenge_input = await page.query_selector('input[name="challenge_response"]')
            if challenge_input:
                print("🔐 Обнаружена двухфакторная аутентификация (2FA).")
                if headless:
                    print("❌ Запустите скрипт с флагом --no-headless для ручного ввода 2FA кода.")
                    print("⚠️ Продолжаем без авторизации")
                    await page.close()
                    return context
                else:
                    print("⏳ Введите 2FA код вручную в браузере и нажмите Enter здесь для продолжения...")
                    input()
        except:
            pass

        # Проверка на ошибки входа
        try:
            error_selectors = [
                'text="Sorry, we could not authenticate you."',
                'text="Неправильное имя пользователя, email или пароль"',
                '[data-testid="LoginForm_Error"]'
            ]
            
            for selector in error_selectors:
                error_element = await page.query_selector(selector)
                if error_element:
                    print("❌ Ошибка входа: неверные учетные данные")
                    print("⚠️ Продолжаем без авторизации")
                    await page.close()
                    return context
        except:
            pass

        # Проверка успешного входа
        success_selectors = [
            'a[aria-label="Profile"]',
            'a[href="/home"]',
            'div[data-testid="primaryColumn"]',
            '[data-testid="SideNav_AccountSwitcher_Button"]'
        ]
        
        for selector in success_selectors:
            try:
                await page.wait_for_selector(selector, timeout=5000)
                print("✅ Успешная авторизация в X.com")
                await page.close()
                return context
            except:
                continue
                
        print("⚠️ Не удалось подтвердить успешный вход, но продолжаем...")
        await page.close()
        return context

    except Exception as e:
        print(f"❌ Ошибка при входе: {e}")
        if not headless:
            print("🔍 Браузер остается открытым для ручной проверки")
            input("Выполните вход вручную (если нужно) и нажмите Enter...")
            await page.close()
            return context
        else:
            print("⚠️ Продолжаем без авторизации")
            await page.close()
            return context

async def check_account(page: Page, username: str) -> Dict:
    """Улучшенная проверка статуса аккаунта с дополнительными тестами"""
    url = f'https://x.com/{username}'
    result = {
        "username": username, 
        "status": "error", 
        "last_activity": None, 
        "external_links": [],
        "follower_count": None,
        "following_count": None,
        "verification": "unknown"
    }
    
    try:
        print(f"🔍 Проверяем аккаунт: {username}")
        await page.goto(url, timeout=20000)
        await asyncio.sleep(random.uniform(2.0, 4.0))
        await human_like_behavior(page)

        # Получение содержимого страницы
        content = await page.content()
        content_lower = content.lower()

        # 1. Проверка на несуществующий аккаунт
        not_exist_phrases = [
            "this account doesn't exist",
            "this page doesn't exist", 
            "этот аккаунт не существует",
            "try searching for another",
            "page does not exist",
            "user not found"
        ]
        
        if any(phrase in content_lower for phrase in not_exist_phrases):
            result['status'] = 'does_not_exist'
            print(f"❌ Аккаунт {username} не существует")
            return result

        # 2. Проверка на заблокированный аккаунт
        if "account suspended" in content_lower:
            result['status'] = 'suspended'
            print(f"🚫 Аккаунт {username} заблокирован")
            return result

        # 3. Проверка на защищенный аккаунт  
        if any(phrase in content_lower for phrase in [
            "these tweets are protected",
            "эти твиты защищены",
            "protected tweets"
        ]):
            result['status'] = 'protected'
            print(f"🔒 Аккаунт {username} защищен")
            return result

        # 4. ОСНОВНАЯ ПРОВЕРКА - попытка кликнуть на Following
        print(f"🔍 Тестируем клик по Following для {username}...")
        
        # Селекторы для поиска элемента "Following"
        following_selectors = [
            'a[href$="/following"]',
            'a[href*="/following"]', 
            'div[data-testid="UserProfileHeader_Items"] a:has-text("Following")',
            'a:has-text("Following")',
            'span:has-text("Following")',
            'div:has-text("Following")',
            '[data-testid="UserProfileHeader_Items"] a[href*="following"]'
        ]
        
        following_clicked = False
        current_following_count = None
        
        for selector in following_selectors:
            try:
                following_element = await page.wait_for_selector(selector, timeout=3000)
                if following_element:
                    print(f"✅ Найден элемент Following: {selector}")
                    
                    # Попытка извлечь количество подписок из текста
                    try:
                        following_text = await following_element.inner_text()
                        numbers = ''.join(c for c in following_text if c.isdigit())
                        if numbers:
                            current_following_count = int(numbers)
                            print(f"📊 Following: {current_following_count}")
                    except:
                        pass
                    
                    # Пробуем кликнуть
                    await following_element.click()
                    await asyncio.sleep(random.uniform(1.5, 3.0))
                    
                    # Проверяем, что произошел переход
                    current_url = page.url
                    if "/following" in current_url:
                        print(f"✅ Успешный клик по Following для {username}")
                        following_clicked = True
                        
                        # Возвращаемся обратно на профиль
                        await page.goto(url, timeout=15000)
                        await asyncio.sleep(random.uniform(1.0, 2.5))
                        break
                    else:
                        print(f"⚠️ Клик не привел к переходу для {username}")
                        
            except Exception as e:
                continue
        
        result['following_count'] = current_following_count

        # 5. Попытка извлечь количество подписчиков
        try:
            followers_selectors = [
                'a:has-text("Followers")',
                'span:has-text("Followers")', 
                'div:has-text("Followers")',
                '[data-testid="UserProfileHeader_Items"] a[href*="followers"]'
            ]
            
            for selector in followers_selectors:
                try:
                    followers_element = await page.query_selector(selector)
                    if followers_element:
                        followers_text = await followers_element.inner_text()
                        numbers = ''.join(c for c in followers_text if c.isdigit())
                        if numbers:
                            result['follower_count'] = int(numbers)
                            print(f"📊 Followers: {result['follower_count']}")
                            break
                except:
                    continue
        except:
            pass

        # 6. Проверка верификации
        try:
            verified_selectors = [
                'svg[data-testid="verificationBadge"]',
                '[data-testid="verificationBadge"]',
                'svg[aria-label="Verified account"]'
            ]
            
            for selector in verified_selectors:
                verified_badge = await page.query_selector(selector)
                if verified_badge:
                    result['verification'] = "verified"
                    print(f"✅ Аккаунт {username} верифицирован")
                    break
            else:
                result['verification'] = "not_verified"
        except:
            pass

        # 7. Дополнительные индикаторы активного профиля
        active_profile_indicators = [
            'div[data-testid="UserProfileHeader"]',
            'div[data-testid="UserName"]', 
            'div[data-testid="UserDescription"]',
            'img[data-testid="UserAvatar"]',
            'div[data-testid="primaryColumn"]'
        ]
        
        profile_found = False
        for indicator in active_profile_indicators:
            try:
                element = await page.query_selector(indicator)
                if element:
                    profile_found = True
                    break
            except:
                continue

        # 8. Извлечение внешних ссылок из профиля
        external_links = []
        try:
            # Поиск ссылок в описании профиля и URL поле
            bio_selectors = [
                'div[data-testid="UserDescription"] a[href^="http"]',
                'div[data-testid="UserDescription"] a[href^="https"]',
                'div[data-testid="UserUrl"] a[href^="http"]',
                'div[data-testid="UserUrl"] a[href^="https"]',
                'span[data-testid="UserUrl"] a',
                'a[data-testid="UserUrl"]'
            ]
            
            for selector in bio_selectors:
                try:
                    links = await page.query_selector_all(selector)
                    for link in links:
                        href = await link.get_attribute('href')
                        if href and href not in external_links and href != url:
                            # Фильтруем только внешние ссылки (не x.com/twitter.com)
                            if not any(domain in href.lower() for domain in ['x.com', 'twitter.com', 't.co']):
                                external_links.append(href)
                                print(f"🔗 Найдена внешняя ссылка: {href}")
                except:
                    continue
                    
        except Exception as e:
            print(f"⚠️ Ошибка извлечения ссылок: {e}")

        result['external_links'] = external_links

        # 9. Попытка извлечь дату последней активности
        try:
            time_element = await page.query_selector('time[datetime]')
            if time_element:
                datetime_str = await time_element.get_attribute('datetime')
                if datetime_str:
                    result['last_activity'] = datetime_str
                    print(f"📅 Последняя активность: {datetime_str}")
        except Exception:
            pass

        # 10. Определение финального статуса
        if following_clicked:
            result['status'] = 'exists_verified'  # Подтверждено кликом
            print(f"✅ Аккаунт {username} подтвержден (клик по Following успешен)")
        elif profile_found:
            result['status'] = 'exists_likely'     # Вероятно существует
            print(f"☑️ Аккаунт {username} вероятно существует (найдены элементы профиля)")
        else:
            result['status'] = 'requires_auth'     # Требует авторизации
            print(f"🔑 Аккаунт {username} требует авторизации")

        return result

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        print(f"⚠️ Ошибка проверки {username}: {e}")
        return result

async def worker(sem: asyncio.Semaphore, context: BrowserContext, username: str) -> Dict:
    """Воркер для обработки одного аккаунта с ограничением конкурентности"""
    async with sem:
        # Случайная задержка между запросами для имитации человеческого поведения
        await asyncio.sleep(random.uniform(1.0, 3.0))
        
        page = await context.new_page()
        try:
            result = await check_account(page, username)
            return result
        finally:
            await page.close()
            # Дополнительная задержка после закрытия страницы
            await asyncio.sleep(random.uniform(0.5, 1.5))

def read_usernames_from_file(filepath: str) -> list:
    """Читает имена пользователей из файла (поддерживает CSV и TXT)"""
    usernames = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as infile:
            # Определяем формат файла
            if filepath.endswith('.csv'):
                reader = csv.DictReader(infile)
                
                # Проверяем наличие колонки "username"
                if 'username' not in reader.fieldnames:
                    print(f"❌ В CSV файле отсутствует обязательная колонка 'username'")
                    print(f"Доступные колонки: {reader.fieldnames}")
                    sys.exit(1)
                
                # Читаем только колонку username
                for row in reader:
                    username = row['username'].strip()
                    if username:
                        # Нормализация имени пользователя
                        if username.startswith('@'):
                            username = username[1:]
                        usernames.append(username)
            else:
                # Обычный текстовый файл - одно имя на строку
                for line in infile:
                    username = line.strip()
                    if username:
                        # Нормализация имени пользователя
                        if username.startswith('@'):
                            username = username[1:]
                        usernames.append(username)

    except Exception as e:
        print(f"❌ Ошибка чтения файла: {e}")
        sys.exit(1)
    
    return usernames

async def main():
    parser = argparse.ArgumentParser(description="Проверка статуса аккаунтов X.com с использованием async Playwright.")
    parser.add_argument('--input', '-i', required=True, help='Входной файл с именами пользователей (CSV с колонкой "username" или TXT)')
    parser.add_argument('--output', '-o', default='results.csv', help='Выходной CSV файл (по умолчанию: results.csv)')
    parser.add_argument('--login', type=str, help='Логин X (имя пользователя/email)')
    parser.add_argument('--password', type=str, help='Пароль X')
    parser.add_argument('--max-concurrent', '-c', type=int, default=3, help='Максимальное количество одновременных проверок (по умолчанию: 3)')
    parser.add_argument('--no-headless', action='store_true', help='Запустить браузер видимым (для отладки и ручной 2FA)')
    
    args = parser.parse_args()

    # Получение учетных данных
    login_cred = args.login or os.getenv('X_LOGIN')
    password_cred = args.password or os.getenv('X_PASSWORD')

    if (login_cred and not password_cred) or (password_cred and not login_cred):
        print("❌ Необходимо указать и логин, и пароль вместе.")
        sys.exit(1)

    # Проверка существования входного файла
    if not os.path.exists(args.input):
        print(f"❌ Входной файл не найден: {args.input}")
        sys.exit(1)

    # Чтение списка пользователей
    usernames = read_usernames_from_file(args.input)

    if not usernames:
        print("❌ Не найдено имен пользователей в файле.")
        sys.exit(1)

    print(f"📊 Найдено {len(usernames)} пользователей для проверки")

    # Запуск Playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not args.no_headless)
        
        # Создание контекста с человекоподобными настройками
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
            viewport={'width': random.randint(1280, 1366), 'height': random.randint(720, 800)},
            locale='en-US',
            timezone_id='America/New_York'
        )

        # Авторизация, если предоставлены учетные данные
        if login_cred and password_cred:
            print("🔐 Выполнение входа в X.com...")
            context = await login_x(context, not args.no_headless, login_cred, password_cred)
        
        # Убеждаемся что context не None
        if context is None:
            print("❌ Критическая ошибка: не удалось создать контекст браузера")
            await browser.close()
            sys.exit(1)

        # Создание семафора для ограничения конкурентности
        sem = asyncio.Semaphore(args.max_concurrent)
        
        print(f"🚀 Начинаем проверку с максимальной конкурентностью: {args.max_concurrent}")

        # Создание задач для всех пользователей
        tasks = [asyncio.create_task(worker(sem, context, username)) for username in usernames]
        results = []

        # Обработка результатов по мере завершения
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            try:
                result = await task
                results.append(result)
                status_emoji = {
                    'exists_verified': '✅',      # Подтверждено кликом
                    'exists_likely': '☑️',       # Вероятно существует
                    'does_not_exist': '❌',
                    'suspended': '🚫',
                    'protected': '🔒',
                    'requires_auth': '🔑',
                    'error': '⚠️'
                }.get(result['status'], '❓')
                
                # Формирование дополнительной информации
                extra_info = []
                if result.get('following_count') is not None:
                    extra_info.append(f"Following: {result['following_count']}")
                if result.get('follower_count') is not None:
                    extra_info.append(f"Followers: {result['follower_count']}")
                if result.get('external_links'):
                    extra_info.append(f"Links: {len(result['external_links'])}")
                if result.get('verification') == 'verified':
                    extra_info.append("✓")
                
                extra_str = f" ({', '.join(extra_info)})" if extra_info else ""
                
                print(f"{status_emoji} [{i}/{len(usernames)}] {result['username']} -> {result['status']}{extra_str}")
                
            except Exception as e:
                print(f"⚠️ [{i}/{len(usernames)}] Ошибка задачи: {e}")
                results.append({"username": "unknown", "status": "error", "last_activity": None, "external_links": [], "following_count": None, "follower_count": None, "verification": "unknown"})

        # Сохранение результатов в CSV
        print(f"💾 Сохранение результатов в {args.output}...")
        with open(args.output, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['username', 'status', 'last_activity', 'external_links', 'following_count', 'follower_count', 'verification'])
            writer.writeheader()
            for result in results:
                # Преобразуем список ссылок в строку
                links_str = '; '.join(result.get('external_links', []))
                writer.writerow({
                    'username': result.get('username', ''),
                    'status': result.get('status', ''),
                    'last_activity': result.get('last_activity', '') or '',
                    'external_links': links_str,
                    'following_count': result.get('following_count', '') or '',
                    'follower_count': result.get('follower_count', '') or '',
                    'verification': result.get('verification', '')
                })

        print(f"✅ Проверка завершена! Результаты сохранены в {args.output}")
        
        # Статистика
        stats = {}
        verified_count = 0
        total_links = 0
        
        for result in results:
            status = result.get('status', 'unknown')
            stats[status] = stats.get(status, 0) + 1
            
            if result.get('verification') == 'verified':
                verified_count += 1
            
            if result.get('external_links'):
                total_links += len(result['external_links'])
        
        print("\n📈 Статистика:")
        for status, count in stats.items():
            emoji = {
                'exists_verified': '✅',
                'exists_likely': '☑️',
                'does_not_exist': '❌', 
                'suspended': '🚫',
                'protected': '🔒',
                'requires_auth': '🔑',
                'error': '⚠️'
            }.get(status, '❓')
            print(f"  {emoji} {status}: {count}")
        
        print(f"\n🔍 Дополнительная статистика:")
        print(f"  ✅ Верифицированных аккаунтов: {verified_count}")
        print(f"  🔗 Всего внешних ссылок найдено: {total_links}")

        await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
