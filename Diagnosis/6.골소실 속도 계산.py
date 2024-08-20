from datetime import datetime

def calculate_average_loss_speed(date_0, date_1, loss_start, loss_end):
    """
    날짜 간격과 골소실률 변화를 기반으로 연평균 골소실 속도를 계산하는 함수입니다.

    :param date_0: 첫 번째 날짜 (YYYY-MM-DD 형식)
    :param date_1: 두 번째 날짜 (YYYY-MM-DD 형식)
    :param loss_start: 첫 번째 날짜의 골소실률 (%)
    :param loss_end: 두 번째 날짜의 골소실률 (%)
    :return: 연평균 골소실 속도 (%/년)
    """
    # 날짜 간격 계산
    start_date = datetime.strptime(date_0, '%Y-%m-%d')
    end_date = datetime.strptime(date_1, '%Y-%m-%d')
    elapsed_days = (end_date - start_date).days
    
    # 골소실률 변화량
    change = loss_end - loss_start
    
    # 연평균 골소실 속도 계산
    average_loss_speed = (change / elapsed_days) * 365
    
    return round(average_loss_speed, 2)

# 예시 데이터
date_0 = '2021-04-19'
date_1 = '2023-12-23'
loss_start = 39
loss_end = 48

# 함수 호출 및 결과 출력
average_loss_speed = calculate_average_loss_speed(date_0, date_1, loss_start, loss_end)
print(f"연평균 골소실 속도: {average_loss_speed}%/year")