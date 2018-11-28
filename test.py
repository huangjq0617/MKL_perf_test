import os

B_MATRIX_NUM = [1, 2, 4, 8]

M = [4, 8, 16, 32, 64]
N = [128, 256, 512, 1024, 2048]
K = [128, 256, 512, 1024, 2048]

loopCountBase = 10000
M_N_K_B_Base = 64 * 1024 * 1024 * 1

fp = open('./test_result.txt', 'w')

for b_num in B_MATRIX_NUM:
    for m in M:
        for n in N:
            for k in K:

                loopCount = loopCountBase * M_N_K_B_Base / m / n / k / b_num
                for i in range(3):
                    cmd = './build/amum_B_{} 0 {} {} {} {}'.format(b_num, m, n, k, loopCount)
                    res = os.popen(cmd).read()
                    print res
                    fp.write(res + '\n')

                print '-' * 40
                fp.write('-' * 40 + '\n')

fp.close()
