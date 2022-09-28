1) 전처리 진행 (preprocess.py)
- INTERVAL_UNIT은 본 데이터의 시간간격을 어떻게 설정할 지 결정하는 변수 (1시간은 '1H', 10분은 '10Min', 10초는 '10s'로 설정, default='1H')
- EXTRA_UNIT은 서로 다른 resolution의 input을 학습할 때 low resolution의 패턴은 빼고 좀 더 high resolution의 input에서 residual pattern만을 학습하기 위한 변수 (default='10Min')
- THIDR_UNIT은 EXTRA_UNIT과 동일 (default='10s')
