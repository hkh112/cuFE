#pragma warning(disable: 4996)

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <gmp.h>

#include "svm/svm.h"
#include "src/sample.h"
#include "src/rlwe_sife.h"

int print_null(const char* s, ...) { return 0; }
static int (*info)(const char* fmt, ...) = &printf;
struct svm_node* x;
int max_nr_attr = 64;
struct svm_model* model;
static char* line = NULL;
static int max_line_len;

#ifdef PERF 
extern void rlwe_sife_keygen_gpu(const uint32_t* y, const uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], uint32_t* sk_y, int repeat, float* part2_time);
extern void rlwe_sife_decrypt_gmp_gpu(uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], const uint32_t* y, uint32_t* sk_y, uint32_t* d_y, int repeat, float* part2_time);
#else
extern void rlwe_sife_keygen_gpu(const uint32_t* y, const uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], uint32_t* sk_y, int repeat);
extern void rlwe_sife_decrypt_gmp_gpu(uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], const uint32_t* y, uint32_t* sk_y, uint32_t* d_y, int repeat);
#endif   

void convertToarray(const struct svm_node* x, unsigned int* m)
{
    int i;
    for (i = 0; i < SIFE_L; i++)
        m[i] = 0;

    while (x->index != -1)
    {
        m[x->index] = x->value;
        ++x;
    }
}

double innerProduct(unsigned int* pm, unsigned int* py)
{
    int i;
    double sum = 0;
    for(i = 0; i < SIFE_L; i++)
    {
        sum += pm[i] * py[i];
    }
    return sum;
}

#ifdef PERF
double svm_predict_plain(const struct svm_model* model, unsigned int* m, double* CLOCK_predict_part1, double* CLOCK_predict_part2)
#else
double svm_predict_plain(const struct svm_model* model, unsigned int* m)
#endif   
{
    int nr_class = model->nr_class; // 모델의 클래스 수
    double* dec_values = Malloc(double, nr_class * (nr_class - 1) / 2);
    int i;
    int l = model->l;               // 모델의 Support Vector 수
    unsigned int y[SIFE_L];         // 일반 배열형 변수 for SV

#ifdef PERF 
    uint64_t CLOCK1, CLOCK2;
    CLOCK1=cpucycles();
#endif   
    double* kvalue = Malloc(double, l);
    for (i = 0; i < l; i++)
    {
        convertToarray(model->SV[i], y);
        kvalue[i] = innerProduct(m, y);
    }
#ifdef PERF 
    CLOCK2=cpucycles();
    *CLOCK_predict_part1 += (double)CLOCK2 - CLOCK1;
#endif   

#ifdef PERF 
    CLOCK1=cpucycles();
#endif   
    int* start = Malloc(int, nr_class);
    start[0] = 0;
    for (i = 1; i < nr_class; i++)
        start[i] = start[i - 1] + model->nSV[i - 1];

    int* vote = Malloc(int, nr_class);
    for (i = 0; i < nr_class; i++)
        vote[i] = 0;

    int p = 0;
    for (i = 0; i < nr_class; i++)
        for (int j = i + 1; j < nr_class; j++)
        {
            double sum = 0;
            int si = start[i];
            int sj = start[j];
            int ci = model->nSV[i];
            int cj = model->nSV[j];

            int k;
            double* coef1 = model->sv_coef[j - 1];
            double* coef2 = model->sv_coef[i];
            for (k = 0; k < ci; k++)
                sum += coef1[si + k] * kvalue[si + k];
            for (k = 0; k < cj; k++)
                sum += coef2[sj + k] * kvalue[sj + k];
            sum -= model->rho[p];
            dec_values[p] = sum;

            if (dec_values[p] > 0)
                ++vote[i];
            else
                ++vote[j];
            p++;
        }

    int vote_max_idx = 0;
    for (i = 1; i < nr_class; i++)
        if (vote[i] > vote[vote_max_idx])
            vote_max_idx = i;
#ifdef PERF 
    CLOCK2=cpucycles();
    *CLOCK_predict_part2 += (double)CLOCK2 - CLOCK1;
#endif   

    free(dec_values);
    free(kvalue);
    free(start);
    free(vote);
    return model->label[vote_max_idx];
}

#ifdef PERF 
double svm_predict_cpu(const struct svm_model* model, uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], const uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], double* CLOCK_predict_part1, double* CLOCK_predict_part2, double* CLOCK_kp, double* CLOCK_dec, double* CLOCK_extract)
#else
double svm_predict_cpu(const struct svm_model* model, uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], const uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N])
#endif   
{
    int nr_class = model->nr_class; // 모델의 클래스 수
    double* dec_values = Malloc(double, nr_class * (nr_class - 1) / 2);
    int i, j;
    int l = model->l;               // 모델의 Support Vector 수
    unsigned int y[SIFE_L];         // 일반 배열형 변수 for SV       
    uint32_t sk_y[SIFE_NMODULI][SIFE_N];
    mpz_t dy[SIFE_N];

    double sum;
    int si;
    int sj;
    int ci;
    int cj;
    int k;
    double* coef1;
    double* coef2;
    double* kvalue = Malloc(double, l);

#ifdef PERF 
    uint64_t CLOCK1, CLOCK2;
#endif   

    for (i = 0; i < l; i++)
    {
#ifdef PERF 
        CLOCK1=cpucycles();
#endif   
        for(j=0;j<SIFE_N;j++){
            mpz_init(dy[j]);
        }
        convertToarray(model->SV[i], y);
#ifdef PERF 
        CLOCK2=cpucycles();
        *CLOCK_predict_part1 += (CLOCK2-CLOCK1);
#endif   

        //Generation of the key for decrypting m·y
#ifdef PERF 
        rlwe_sife_keygen(y, msk, sk_y, CLOCK_kp);
#else
        rlwe_sife_keygen(y, msk, sk_y);
#endif   

        //Decryption of m·y
#ifdef PERF 
        rlwe_sife_decrypt_gmp(c, y, sk_y, dy, CLOCK_dec);
#else
        rlwe_sife_decrypt_gmp(c, y, sk_y, dy);
#endif   

        //Extraction of the result (cancel scaling)
#ifdef PERF 
        CLOCK1=cpucycles();
        round_extract_gmp(dy);      
        CLOCK2=cpucycles();
        *CLOCK_extract += (CLOCK2-CLOCK1);
#else
        round_extract_gmp(dy);      
#endif   

#ifdef PERF 
        CLOCK1=cpucycles();
#endif   
        kvalue[i] = mpz_get_d(dy[0]);
#ifdef PERF 
        CLOCK2=cpucycles();
        *CLOCK_predict_part1 += (CLOCK2-CLOCK1);
#endif   
    }

#ifdef PERF 
        CLOCK1=cpucycles();
#endif   
    int* start = Malloc(int, nr_class);
    start[0] = 0;
    for (i = 1; i < nr_class; i++)
        start[i] = start[i - 1] + model->nSV[i - 1];

    int* vote = Malloc(int, nr_class);
    for (i = 0; i < nr_class; i++)
        vote[i] = 0;

    int p = 0;
    for (i = 0; i < nr_class; i++)
        for (j = i + 1; j < nr_class; j++)
        {
            sum = 0;
            si = start[i];
            sj = start[j];
            ci = model->nSV[i];
            cj = model->nSV[j];

            coef1 = model->sv_coef[j - 1];
            coef2 = model->sv_coef[i];
            for (k = 0; k < ci; k++)
                sum += coef1[si + k] * kvalue[si + k];
            for (k = 0; k < cj; k++)
                sum += coef2[sj + k] * kvalue[sj + k];
            sum -= model->rho[p];
            dec_values[p] = sum;

            if (dec_values[p] > 0)
                ++vote[i];
            else
                ++vote[j];
            p++;
        }

    int vote_max_idx = 0;
    for (i = 1; i < nr_class; i++)
        if (vote[i] > vote[vote_max_idx])
            vote_max_idx = i;
#ifdef PERF 
        CLOCK2=cpucycles();
        *CLOCK_predict_part2 += (CLOCK2-CLOCK1);
#endif   

    for(j=0;j<SIFE_N;j++){
        mpz_clear(dy[j]);
    }
    free(dec_values);
    free(kvalue);
    free(start);
    free(vote);
    return model->label[vote_max_idx];
}

#ifdef PERF 
double svm_predict_gpu(const struct svm_model* model, uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], const uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], double* CLOCK_predict_part1, double* CLOCK_predict_part2,  double* kp_part1_time, float* kp_part2_time, double* dec_part1_time, float* dec_part2_time, double* extract_part1_time)
#else
double svm_predict_gpu(const struct svm_model* model, uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], const uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N])
#endif   
{
    int nr_class = model->nr_class; // 모델의 클래스 수
    double* dec_values = Malloc(double, nr_class * (nr_class - 1) / 2);
    int i, j;
    int l = model->l;               // 모델의 Support Vector 수
    unsigned int* y = Malloc(unsigned int, l * SIFE_L);         // 일반 배열형 변수 for SV       
    uint32_t* sk_y = Malloc(uint32_t, l * SIFE_NMODULI * SIFE_N);
    uint32_t* d_y = Malloc(uint32_t, l * SIFE_NMODULI * SIFE_N);
    uint32_t dy[SIFE_NMODULI][SIFE_N];

    double sum;
    int si, sj, ci, cj, k;
    double* coef1;
    double* coef2;
    double* kvalue = Malloc(double, l);

#ifdef PERF 
    uint64_t CLOCK1, CLOCK2;
#endif   

    for (i = 0; i < l; i++)
    {
#ifdef PERF 
        CLOCK1=cpucycles();
#endif
        convertToarray(model->SV[i], y + i * SIFE_L);
#ifdef PERF 
        CLOCK2=cpucycles();
        *CLOCK_predict_part1 += (CLOCK2-CLOCK1);
#endif   
    }

    //Generation of the key for decrypting m·y
#ifdef PERF 
    rlwe_sife_keygen_gpu(y, msk, sk_y, l, kp_part2_time);
#else
    rlwe_sife_keygen_gpu(y, msk, sk_y, l);
#endif   

    //Decryption of m·y
#ifdef PERF 
    rlwe_sife_decrypt_gmp_gpu(c, y, sk_y, d_y, l, dec_part2_time);
#else
    rlwe_sife_decrypt_gmp_gpu(c, y, sk_y, d_y, l);
#endif   

    //Extraction of the result (cancel scaling)
    for (i = 0; i < l; i++)
    {
#ifdef PERF 
        CLOCK1=cpucycles();
        memcpy(dy, d_y + i * SIFE_NMODULI * SIFE_N, SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
        kvalue[i] = round_extract_gmp2(dy);      
        CLOCK2=cpucycles();
        *extract_part1_time += (CLOCK2-CLOCK1);
#else
        memcpy(dy, d_y + i * SIFE_NMODULI * SIFE_N, SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
        kvalue[i] = round_extract_gmp2(dy);   
#endif   
    }

#ifdef PERF 
        CLOCK1=cpucycles();
#endif   
    int* start = Malloc(int, nr_class);
    start[0] = 0;
    for (i = 1; i < nr_class; i++)
        start[i] = start[i - 1] + model->nSV[i - 1];

    int* vote = Malloc(int, nr_class);
    for (i = 0; i < nr_class; i++)
        vote[i] = 0;

    int p = 0;
    for (i = 0; i < nr_class; i++)
        for (j = i + 1; j < nr_class; j++)
        {
            sum = 0;
            si = start[i];
            sj = start[j];
            ci = model->nSV[i];
            cj = model->nSV[j];

            coef1 = model->sv_coef[j - 1];
            coef2 = model->sv_coef[i];
            for (k = 0; k < ci; k++)
                sum += coef1[si + k] * kvalue[si + k];
            for (k = 0; k < cj; k++)
                sum += coef2[sj + k] * kvalue[sj + k];
            sum -= model->rho[p];
            dec_values[p] = sum;

            if (dec_values[p] > 0)
                ++vote[i];
            else
                ++vote[j];
            p++;
        }

    int vote_max_idx = 0;
    for (i = 1; i < nr_class; i++)
        if (vote[i] > vote[vote_max_idx])
            vote_max_idx = i;
#ifdef PERF 
        CLOCK2=cpucycles();
        *CLOCK_predict_part2 += (CLOCK2-CLOCK1);
#endif   

    free(y);
    free(sk_y);
    free(d_y);
    free(dec_values);
    free(kvalue);
    free(start);
    free(vote);
    return model->label[vote_max_idx];
}

static char* readline(FILE* input)
{
    int len;

    if (fgets(line, max_line_len, input) == NULL)
        return NULL;

    while (strrchr(line, '\n') == NULL)
    {
        max_line_len *= 2;
        line = (char*)realloc(line, max_line_len);
        len = (int)strlen(line);
        if (fgets(line + len, max_line_len - len, input) == NULL)
            break;
    }
    return line;
}

void exit_input_error(int line_num)
{
    fprintf(stderr, "Wrong input format at line %d\n", line_num);
    exit(1);
}

void predict(FILE* input, FILE* output)
{
    int correct = 0;
    int correct2 = 0;
    int correct3 = 0;
    int total = 0;
    double error = 0;
    double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

    int svm_type = svm_get_svm_type(model);
    int nr_class = svm_get_nr_class(model);
    double* prob_estimates = NULL;
    int i=0, j=0, n=0;
    int g=0;
    double target_label[max_data], predict_label[max_data], predict_label2[max_data], predict_label3[max_data];

    // Declarate variables
    uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N];
    uint32_t mpk2[SIFE_L+1][SIFE_NMODULI][SIFE_N];
    uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N];
    uint32_t msk2[SIFE_L][SIFE_NMODULI][SIFE_N];
    uint32_t c[max_data][SIFE_L+1][SIFE_NMODULI][SIFE_N];
    uint32_t c2[max_data][SIFE_L+1][SIFE_NMODULI][SIFE_N];
    uint32_t m[max_data][SIFE_L];

    uint64_t CLOCK1, CLOCK2;
    double CLOCK_read, CLOCK_su, CLOCK_enc, CLOCK_predict, CLOCK_kp, CLOCK_dec, CLOCK_extract;
    double CLOCK_predict1, CLOCK_predict1_part1, CLOCK_predict1_part2;
    double CLOCK_predict2, CLOCK_predict2_part1, CLOCK_predict2_part2;
    double CLOCK_predict3, CLOCK_predict3_part1, CLOCK_predict3_part2;

    double setup_part1_time=0, enc_part1_time=0, kp_part1_time=0, dec_part1_time=0, extract_part1_time=0;
    float setup_part2_time=0, enc_part2_time=0, kp_part2_time=0, dec_part2_time=0;

    CLOCK1 = CLOCK2 = 0;
    CLOCK_read = CLOCK_su = CLOCK_enc = CLOCK_predict = CLOCK_kp = CLOCK_dec = CLOCK_extract = 0;
    CLOCK_predict1 = CLOCK_predict1_part1 = CLOCK_predict1_part2 = 0;
    CLOCK_predict2 = CLOCK_predict2_part1 = CLOCK_predict2_part2 = 0;
    CLOCK_predict3 = CLOCK_predict3_part1 = CLOCK_predict3_part2 = 0;


    max_line_len = 1024;
    line = (char*)malloc(max_line_len * sizeof(char));
#ifdef PERF 
    CLOCK1=cpucycles();
#endif   
    while (readline(input) != NULL)
    {
        g++;
        if (g <= 100)
            continue;

        i = 0;
        char* idx, * val, * label, * endptr;
        int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

        label = strtok(line, " \t\n");
        if (label == NULL) // empty line
            exit_input_error(total + 1);

        // 데이터 추출(레이블)
        target_label[n] = strtod(label, &endptr);
        if (endptr == label || *endptr != '\0')
            exit_input_error(total + 1);

        // 데이터 추출(데이터)
        while (1)
        {
            if (i >= max_nr_attr - 1)   // need one more for index = -1
            {
                max_nr_attr *= 2;
                x = (struct svm_node*)realloc(x, max_nr_attr * sizeof(struct svm_node));
            }

            idx = strtok(NULL, ":");
            val = strtok(NULL, " \t");

            if (val == NULL)
                break;
            errno = 0;
            x[i].index = (int)strtol(idx, &endptr, 10);
            if (endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
                exit_input_error(total + 1);
            else
                inst_max_index = x[i].index;

            errno = 0;
            x[i].value = strtod(val, &endptr);
            if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                exit_input_error(total + 1);

            ++i;
        }
        x[i].index = -1;

        convertToarray(x, m[n]);

        n++;
        // 데이터가 너무 많아서 제한함
        if (n >= max_data)
            break;
    }
#ifdef PERF 
    CLOCK2=cpucycles();
    CLOCK_read += (CLOCK2-CLOCK1);
#endif   

    //Generation of master secret key sk and master public key pk pair
    // 키 생성
#ifdef PERF 
    for(i=0;i<max_data;i++){
        #ifdef CPU
            rlwe_sife_setup(mpk, msk, &CLOCK_su);
        #endif
        #ifdef GPU
            rlwe_sife_setup_gui(mpk2, msk2, &setup_part1_time, &setup_part2_time);
        #endif
    }
#else
    #ifdef CPU
        rlwe_sife_setup(mpk, msk);
    #endif
    #ifdef GPU
        rlwe_sife_setup_gui(mpk2, msk2);
    #endif
#endif   

    // Encryption of the message m
#ifdef PERF 
    #ifdef CPU
        for(i=0;i<max_data;i++){
            rlwe_sife_encrypt(m[i], mpk, c[i], &CLOCK_enc);
        }
    #endif
    #ifdef GPU
        rlwe_sife_encrypt_gui((uint32_t*)m, mpk2, (uint32_t*)c2, n, &enc_part1_time, &enc_part2_time);
    #endif
#else
    #ifdef CPU
        for(i=0;i<max_data;i++){
            rlwe_sife_encrypt(m[i], mpk, c[i]);
        }
    #endif
    #ifdef GPU
        rlwe_sife_encrypt_gui((uint32_t*)m, mpk2, (uint32_t*)c2, n);
    #endif
#endif   
    //printf("Encryption done \n");

    for(i=0; i<n; i++)
    {
        // 기본 버전 predict2
#ifdef PERF 
        predict_label[i] = svm_predict_plain(model, m[i], &CLOCK_predict1_part1, &CLOCK_predict1_part2);
#else
        predict_label[i] = svm_predict_plain(model, m[i]);
#endif   

        // 암호화 버전 predict3
    #ifdef CPU
#ifdef PERF 
        predict_label2[i] = svm_predict_cpu(model, c[i], msk, &CLOCK_predict2_part1, &CLOCK_predict2_part2, &CLOCK_kp, &CLOCK_dec, &CLOCK_extract);
#else
        predict_label2[i] = svm_predict_cpu(model, c[i], msk);
#endif   
    #endif

        // 병렬 암호화 버전 predict4
    #ifdef GPU
#ifdef PERF 
        predict_label3[i] = svm_predict_gpu(model, c2[i], msk2, &CLOCK_predict3_part1, &CLOCK_predict3_part2, &kp_part1_time, &kp_part2_time, &dec_part1_time, &dec_part2_time, &extract_part1_time);
#else
        predict_label3[i] = svm_predict_gpu(model, c2[i], msk2);
#endif   
    #endif

        // Functional verification
    #ifdef CPU
        if(predict_label[i] != predict_label2[i])
            printf("The %dth IPFE(cpu) result is not correct\n", i+1);
    #endif
    #ifdef GPU
        if(predict_label[i] != predict_label3[i])
            printf("The %dth IPFE(gpu) result is not correct\n", i+1);
    #endif

        // 평가
        if (predict_label[i] == target_label[i])
            ++correct;
    #ifdef CPU
        if (predict_label2[i] == target_label[i])
            ++correct2;
    #endif
    #ifdef GPU
        if (predict_label3[i] == target_label[i])
            ++correct3;
    #endif
        error += (predict_label[i] - target_label[i]) * (predict_label[i] - target_label[i]);
        sump += predict_label[i];
        sumt += target_label[i];
        sumpp += predict_label[i] * predict_label[i];
        sumtt += target_label[i] * target_label[i];
        sumpt += predict_label[i] * target_label[i];
        ++total;
    }
    printf("TEST %d DONE!\n\n", n);

#ifdef AVX2
    printf("AVX2 on\n");
#else
    printf("AVX2 off\n");
#endif

#ifdef PERF 
    printf("<Original SVM>\n");
        info("Accuracy = %g%% (%d/%d) (classification)\n", (double)correct / total * 100, correct, total);
    printf("Average times data preparation: \t \t \t %.6f ms\n", CLOCK_read/CLOCKS_PER_MS/total);
    printf("Average times prediction: \t \t \t \t %.6f ms\n", (CLOCK_predict1_part1 + CLOCK_predict1_part2)/CLOCKS_PER_MS/total);
    printf("   Average times prediction ip part: \t \t \t %.6f ms\n", CLOCK_predict1_part1/CLOCKS_PER_MS/total);
    printf("   Average times prediction decission part: \t \t %.6f ms\n", CLOCK_predict1_part2/CLOCKS_PER_MS/total);
    printf("Average times total time: \t \t \t \t %.6f ms\n", (CLOCK_read + CLOCK_predict1_part1 + CLOCK_predict1_part2)/CLOCKS_PER_MS/total);
    printf("\n"); 

    #ifdef CPU
    printf("<SVM with CPU IPFE>\n");
        info("Accuracy = %g%% (%d/%d) (classification)\n", (double)correct2 / total * 100, correct2, total);
    printf("Average times data preparation: \t \t \t %.6f ms\n", CLOCK_read/CLOCKS_PER_MS/total);
    printf("Average times data encryption: \t \t \t \t %.6f ms\n", (double)CLOCK_su/CLOCKS_PER_MS + CLOCK_enc/CLOCKS_PER_MS/total);
    printf("      Average times data encryption(setup): \t \t %.6f ms\n", (double)CLOCK_su/CLOCKS_PER_MS);
    printf("      Average times data encryption(enc): \t \t %.6f ms\n", (double)CLOCK_enc/CLOCKS_PER_MS/total);
    printf("Average times prediction: \t \t \t \t %.6f ms\n", (CLOCK_predict2_part1 + CLOCK_kp + CLOCK_dec + CLOCK_extract + CLOCK_predict2_part2)/CLOCKS_PER_MS/total);
    printf("   Average times prediction ip part: \t \t \t %.6f ms\n", (CLOCK_predict2_part1 + CLOCK_kp + CLOCK_dec + CLOCK_extract)/CLOCKS_PER_MS/total);
    printf("      Average times prediction ip part(kp): \t \t %.6f ms\n", CLOCK_kp/CLOCKS_PER_MS/total);
    printf("      Average times prediction ip part(dec): \t \t %.6f ms\n", CLOCK_dec/CLOCKS_PER_MS/total);
    printf("      Average times prediction ip part(extract): \t %.6f ms\n", CLOCK_extract/CLOCKS_PER_MS/total);
    printf("      Average times prediction ip part(other): \t \t %.6f ms\n", CLOCK_predict2_part1/CLOCKS_PER_MS/total);
    printf("   Average times prediction2 decission part: \t \t %.6f ms\n", CLOCK_predict2_part2/CLOCKS_PER_MS/total);
    printf("Average times total time: \t \t \t \t %.6f ms\n", (CLOCK_read + CLOCK_su + CLOCK_enc + CLOCK_predict2_part1 + CLOCK_kp + CLOCK_dec + CLOCK_extract + CLOCK_predict2_part2)/CLOCKS_PER_MS/total);
    printf("\n"); 
    #endif
    #ifdef GPU
    printf("<SVM with GPU IPFE>\n");
        info("Accuracy = %g%% (%d/%d) (classification)\n", (double)correct3 / total * 100, correct3, total);
    printf("Average times data preparation: \t \t \t %.6f ms\n", CLOCK_read/CLOCKS_PER_MS/total);
    printf("Average times data encryption: \t \t \t \t %.6f ms\n", (double)setup_part1_time/CLOCKS_PER_MS + setup_part2_time + (enc_part1_time/CLOCKS_PER_MS + enc_part2_time)/total);
    printf("      Average times data encryption(setup): \t \t %.6f ms\n", (double)(setup_part1_time/CLOCKS_PER_MS + setup_part2_time)/total);
    printf("         Average times data encryption cpu(setup): \t %.6f ms\n", (double)setup_part1_time/CLOCKS_PER_MS/total);
    printf("         Average times data encryption gpu(setup): \t %.6f ms\n", (double)setup_part2_time/total);
    printf("      Average times data encryption(enc): \t \t %.6f ms\n", (double)(enc_part1_time/CLOCKS_PER_MS + enc_part2_time)/total);
    printf("         Average times data encryption cpu(enc): \t %.6f ms\n", (double)enc_part1_time/CLOCKS_PER_MS/total);
    printf("         Average times data encryption gpu(enc): \t %.6f ms\n", (double)enc_part2_time/total);
    printf("Average times prediction: \t \t \t \t %.6f ms\n", (double)((CLOCK_predict3_part1 + kp_part1_time + dec_part1_time + extract_part1_time + CLOCK_predict3_part2)/CLOCKS_PER_MS + kp_part2_time + dec_part2_time)/total);
    printf("   Average times prediction ip part: \t \t \t %.6f ms\n", (double)((CLOCK_predict3_part1 + kp_part1_time + dec_part1_time + extract_part1_time)/CLOCKS_PER_MS + kp_part2_time + dec_part2_time)/total);
    printf("      Average times prediction ip part(kp): \t \t %.6f ms\n", (double)(kp_part1_time/CLOCKS_PER_MS + kp_part2_time)/total);
    printf("         Average times prediction ip part cpu(kp): \t %.6f ms\n", (double)kp_part1_time/CLOCKS_PER_MS/total);
    printf("         Average times prediction ip part gpu(kp): \t %.6f ms\n", (double)kp_part2_time/total);
    printf("      Average times prediction ip part(dec): \t \t %.6f ms\n", (double)(dec_part1_time/CLOCKS_PER_MS + dec_part2_time)/total);
    printf("         Average times prediction ip part cpu(dec): \t %.6f ms\n", (double)dec_part1_time/CLOCKS_PER_MS/total);
    printf("         Average times prediction ip part gpu(dec): \t %.6f ms\n", (double)dec_part2_time/total);
    printf("      Average times prediction ip part(extract): \t %.6f ms\n", (double)extract_part1_time/CLOCKS_PER_MS/total);
    printf("      Average times prediction ip part(other): \t \t %.6f ms\n", (double)CLOCK_predict3_part1/CLOCKS_PER_MS/total);
    printf("   Average times prediction decission part: \t \t %.6f ms\n", (double)CLOCK_predict3_part2/CLOCKS_PER_MS/total);
    printf("Average times total time: \t \t \t \t %.6f ms\n", (double)((CLOCK_read + setup_part1_time + enc_part1_time + CLOCK_predict3_part1 + kp_part1_time + dec_part1_time + extract_part1_time + CLOCK_predict3_part2)/CLOCKS_PER_MS + setup_part2_time + enc_part2_time + kp_part2_time + dec_part2_time)/total);
    printf("\n"); 
    #endif
#else
    printf("<Original SVM>\n");
        info("Accuracy = %g%% (%d/%d) (classification)\n", (double)correct / total * 100, correct, total);
    printf("<SVM with CPU IPFE>\n");
        info("Accuracy = %g%% (%d/%d) (classification)\n", (double)correct2 / total * 100, correct2, total);
    printf("<SVM with GPU IPFE>\n");
        info("Accuracy = %g%% (%d/%d) (classification)\n", (double)correct3 / total * 100, correct3, total);
#endif   
}

void exit_with_help()
{
    printf(
        "Usage: svm-predict [options] test_file model_file output_file\n"
        "options:\n"
        "-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n"
        "-q : quiet mode (no outputs)\n"
    );
    exit(1);
}

int main(int argc, char** argv)
{
    FILE* input, *output;
    int i;
    // parse options
    for (i = 1; i < argc; i++)
    {
        if (argv[i][0] != '-') break;
        ++i;
        switch (argv[i - 1][1])
        {
        case 'q':
            info = &print_null;
            i--;
            break;
        default:
            fprintf(stderr, "Unknown option: -%c\n", argv[i - 1][1]);
            exit_with_help();
        }
    }

    if (i >= argc - 2)
        exit_with_help();

    input = fopen(argv[i], "r");
    if (input == NULL)
    {
        fprintf(stderr, "can't open input file %s\n", argv[i]);
        exit(1);
    }

    output = fopen(argv[i + 2], "w");
    if (output == NULL)
    {
        fprintf(stderr, "can't open output file %s\n", argv[i + 2]);
        exit(1);
    }

    if ((model = svm_load_model(argv[i + 1])) == 0)
    {
        fprintf(stderr, "can't open model file %s\n", argv[i + 1]);
        exit(1);
    }

    x = (struct svm_node*)malloc(max_nr_attr * sizeof(struct svm_node));
    if (svm_check_probability_model(model) != 0)
        info("Model supports probability estimates, but disabled in prediction.\n");

    predict(input, output);
    svm_free_and_destroy_model(&model);
    free(x);
    free(line);
    fclose(input);
    fclose(output);
    return 0;
}