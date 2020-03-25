import os

B_MATRIX_NUM = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

#M = [4, 8, 16, 32, 64, 128]
M = [256]
N = [128, 256, 512, 1024, 2048]
K = [128, 256, 512, 1024, 2048]

loopCountBase = 1000
M_N_K_B_Base = 32 * 256 * 256 * 100

fp = open('./test_result.txt', 'w')

for b_num in B_MATRIX_NUM:
    for m in M:
        for n in N:
            for k in K:

                loopCount = max(1, loopCountBase * M_N_K_B_Base / m / n / k / b_num)
                for i in range(1):
                    cmd = './build/bin/amun 0 {} {} {} {} {}'.format(m, n, k, loopCount, b_num)
                    res = os.popen(cmd).read()
                    print res
                    fp.write(res + '\n')

                print '-' * 40
                fp.write('-' * 40 + '\n')

fp.close()
