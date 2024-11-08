
# Функция принимает на вход строку, которая
# состоит из скобок трех типов: круглые, квадратные
# и фигурные. Функция должна проверить, является ли
# переданная последовательность скобок сбалансированной
# или нет. Функция возвращает True, если последовательность
# сбалансирована, и False – в противном случае.
def is_balanced(line):
    pairs = {
        ')': '(',
        ']': '[',
        '}': '{'}

    stack = []
    for i in line:
        if i in pairs.values():
            stack.append(i)
        if i in pairs:
            if len(stack) == 0:
                break
            elif pairs[i] == stack[len(stack) - 1]:
                stack.pop()
            else:
                break
    return len(stack) == 0

def test_is_balanced():
    cases = {
        '((((((((())))))))': False,
        '{[()]}{{}}': True,
        '{[()]}{]{}}': False,
        '{{{((([[[]]])))}}}': True,
        '{}': True,
        '(': False,
        '(}': False,
        '(((())))[]{}': True,
        '((()': False,
        '[{}{})(]': False,
        '{[]{([[[[[[]]]]]])}}': True,
        '{[]{([[[[[[]])]]])}}': False,
    }
    for i, case in enumerate(cases.keys()):
        if is_balanced(case) == cases[case]:
            print(f'{i}: OK')
        else:
            print(f'{i}: Not OK')


def main():
    test_is_balanced()


if __name__ == '__main__':
    main()