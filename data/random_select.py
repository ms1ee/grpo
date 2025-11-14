import random
import json

# 재현성을 위해 랜덤 시드를 설정합니다. (숫자는 아무거나 상관없습니다)
random.seed(42)

# 원본 파일 경로와 저장할 파일 경로를 지정합니다.
input_jsonl_path = 'kor_law/all.jsonl'
output_jsonl_path = 'kor_law/test_k_law_2000.jsonl'
number_of_lines_to_select = 2000

try:
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    if len(all_lines) < number_of_lines_to_select:
        print(f"오류: 파일의 전체 줄({len(all_lines)}개)이 추출하려는 수({number_of_lines_to_select}개)보다 적습니다.")
    else:
        # random.seed(42) 설정 때문에 이 부분은 항상 동일한 라인을 선택하게 됩니다.
        random_sample_lines = random.sample(all_lines, number_of_lines_to_select)

        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            f.writelines(random_sample_lines)

        print(f"성공! '{input_jsonl_path}' 파일에서 (고정된 시드로) 임의의 {number_of_lines_to_select}개 줄을 추출하여 '{output_jsonl_path}' 파일로 저장했습니다.")

except FileNotFoundError:
    print(f"오류: '{input_jsonl_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")
