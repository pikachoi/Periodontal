
import os
import subprocess
import sqlite3


def del_records_from_all_table():
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()

    table_name = ['Accounts_doctor',
                  'Accounts_doctor_profile',
                  'Accounts_question',
                  'Diagnosis_crop',
                  'Diagnosis_imagepath',
                  'Diagnosis_patient',
                  'auth_group',
                  'auth_group_permissions',
                  'auth_permission',
                  'django_admin_log',
                  'django_content_type',
                  'django_migrations',
                  'django_session',]

    for i in table_name:
        delete_query = f"DELETE FROM {i}"

        cursor.execute(delete_query)

    conn.commit()
    conn.close()
    print("############################### 완료 ###############################")


def del_migrations_py():
    # base_path 내 모든 migrations 폴더 찾기
    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            if dir == 'migrations':
                migrations_path = os.path.join(root, dir)
                
                # migrations 폴더 내의 파일 목록 얻기
                migration_files = os.listdir(migrations_path)
                
                # .py 확장자를 가진 파일 삭제 (단, __init__.py는 제외)
                for file in migration_files:
                    if file.endswith('.py') and file != '__init__.py':
                        file_path = os.path.join(migrations_path, file)
                        os.remove(file_path)
                        print(f"삭제된 파일: {file_path}")
    print("############################### 완료 ###############################")


def del_migrations_pycache_pyc():
    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            if dir == 'migrations':
                migrations_path = os.path.join(root, dir)
                pycache_path = os.path.join(migrations_path, '__pycache__')
                
                # migrations 폴더 내의 __pycache__ 폴더 내의 .pyc 파일 삭제
                if os.path.exists(pycache_path):
                    for file in os.listdir(pycache_path):
                        if file.endswith('.pyc'):
                            file_path = os.path.join(pycache_path, file)
                            os.remove(file_path)
                            print(f"삭제된 파일: {file_path}")
        
    print("############################### 완료 ###############################")


def del_pycache_path_pyc():
    # 프로젝트의 모든 __pycache__ 폴더 찾기
    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            if dir == '__pycache__':
                pycache_path = os.path.join(root, dir)
                
                # __pycache__ 폴더 내의 파일 목록 얻기
                migration_files = os.listdir(pycache_path)
                
                # .pyc 확장자를 가진 파일 삭제
                for file in migration_files:
                    if file.endswith('.pyc'):
                        file_path = os.path.join(pycache_path, file)
                        os.remove(file_path)
                        print(f"삭제된 파일: {file_path}")

    print("############################### 완료 ###############################")


def del_sqlite3():
    command = "del db.sqlite3"  # 리눅스: rm db.sqlite3
    subprocess.run(command, shell=True)
    print("############################### 완료 ###############################")


def makemigrations():
    command = "python manage.py makemigrations"
    subprocess.run(command, shell=True)
    print("############################### 완료 ###############################")


def migrate():
    command = "python manage.py migrate"  
    subprocess.run(command, shell=True)
    print("############################### 완료 ###############################")


def append_question():
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    questions_to_insert = [
        ("기억에 남는 추억의 장소는?",),
        ("자신의 인생 좌우명은?",),
        ("자신의 보물 제1호는?",),
        ("추억하고 싶은 날짜다 있다면?",),
        ("다시 태어나면 되고 싶은 것은?",),
        ("인상 깊게 읽은 책 이름은?",),
        ("받았던 선물 중 기억에 남는 독특한 선물은?",),
        ("초등학교 때 별명은?",),
    ]

    for question in questions_to_insert:
        insert_query = "INSERT INTO Accounts_question (question) VALUES (?)"
        cursor.execute(insert_query, question)
    
    conn.commit()
    conn.close()
    print("############################### 완료 ###############################")


if __name__=="__main__":
    # 기본 경로 설정
    base_path = "D:\LT4\PERIO_24.03.05\WebPlatform"

    while True:
        print("------------------------메뉴--------------------------")
        print("-----------------신중하게 메뉴 입력할것!-----------------")

        print("0: 장고 모든 모델 레코드 초기화")
        print("1: migrations 폴더 하위 __init__.py 제외 모든 .py 파일 삭제")
        print("2: migrations/__pycache__ 폴더 하위 모든 .pyc 파일 삭제")
        print("3: 프로젝트의 모든 __pycache__ 폴더 하위 .pyc 파일 삭제")
        print("4: db.sqlite3 삭제")
        print("5: 데이터베이스 모델의 변경 사항을 추적하고 이를 마이그레이션 파일로 생성")
        print("6: 마이그레이션 파일로 데이터베이스 스키마를 업데이트")
        print("7: 질문 추가")
        print("q: 종료")
        
        print("----------------메뉴 번호를 입력하세요.----------------")
        print("번호 입력 =")
        key = input()

        if key == '0':
            del_records_from_all_table()
        elif key == '1':
            del_migrations_py()
        elif key == '2':
            del_migrations_pycache_pyc()
        elif key == '3':
            del_pycache_path_pyc()
        elif key == '4':
            del_sqlite3()
        elif key == '5':
            makemigrations()
        elif key == '6':
            migrate()
        elif key == '7':
            append_question()
        
        elif key == 'q':
            break
        else:
            print('키 입력 에러')