#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "c_cat_mod_flipflop.h"
#include "c_crf_flipflop.h"

#define _DEFAULT_NBASE 4
#define LARGE_VAL 1e30f


static inline float logsumexpf(float x, float y, float a){
    return fmaxf(x, y) + log1pf(expf(-a * fabsf(x-y))) / a;
}

static inline size_t nstate_to_nbase(size_t ntrans_state){
    double nbase_d = sqrt(0.25 + (0.5 * ntrans_state)) - 0.5;
    assert(fmod(nbase_d, 1.0) == 0.0);
    return (size_t) round(nbase_d);
}


/*
******************************
Training/groud truth functions
******************************
 */


void cm_flipflop_forward_step(float const * logprob, float const * fwdprev,
                              int32_t const * seq, size_t nseqpos,
                              float * fwdcurr, float sharpfact,
                              size_t nbase, int32_t const * mod_cats,
                              int32_t const * can_mods_offsets,
                              float const * mod_cat_weights,
                              float mod_weight){
    assert(nseqpos > 0);
    assert(NULL != logprob);
    assert(NULL != fwdprev);
    assert(NULL != seq);
    assert(NULL != fwdcurr);
    assert(NULL != mod_cats);
    assert(NULL != can_mods_offsets);

    assert(nbase > 0);
    const size_t offset_stay = nbase * (nbase + nbase);
    const size_t offset_mod = offset_stay + nbase + nbase;

    for(size_t pos=0 ; pos < nseqpos ; pos++){
        // Stay in current position
        const size_t base = seq[pos];
        fwdcurr[pos] = (base < nbase) ?
          logprob[base * (nbase + nbase) + base]:
          logprob[offset_stay + base];
        fwdcurr[pos] += fwdprev[pos];
    }
    for(size_t pos=1 ; pos < nseqpos ; pos++){
        // Move to new position
        const size_t base_to = seq[pos];
        const size_t base_from = seq[pos - 1];
        size_t can_base_to = base_to;
        if(can_base_to >= nbase){
            can_base_to -= nbase;
        }

        assert(base_to != base_from);  // Can't have repeated bases
        assert(base_to < nbase || base_from + nbase == base_to);
        const float can_score = (base_to < nbase) ?
            logprob[base_to * (nbase + nbase) + base_from] :
            logprob[offset_stay + base_from];
        const size_t mod_idx = can_mods_offsets[can_base_to] + mod_cats[pos];
        const float mod_score =
            mod_weight * mod_cat_weights[mod_idx] *
            logprob[offset_mod + mod_idx];
        fwdcurr[pos] = logsumexpf(fwdcurr[pos],
                                  fwdprev[pos - 1] + can_score + mod_score,
                                  sharpfact);
    }
}


float cm_flipflop_forward(float const * logprob, size_t nblk, size_t ldp,
                          int32_t const * seq, size_t nseqpos,
                          float sharpfact, float * fwd, size_t nbase,
                          int32_t const * mod_cats,
                          int32_t const * can_mods_offsets,
                          float const * mod_cat_weights,
                          float mod_weight){
    assert(nseqpos > 0);
    assert(NULL != logprob);
    assert(NULL != seq);
    assert(NULL != fwd);

    //  Point prior  -- must start in stay at beginning of sequence
    for(size_t pos=0 ; pos < nseqpos ; pos++){
        fwd[pos] = -LARGE_VAL;
    }
    fwd[0] = 0.0;

    for(size_t blk=0 ; blk < nblk ; blk++){
        float const * fwdprev = fwd + blk * nseqpos;
        float * fwdcurr = fwd + (blk + 1) * nseqpos;
        float const * logprobcurr = logprob + blk * ldp;

        cm_flipflop_forward_step(logprobcurr, fwdprev, seq, nseqpos,
                                 fwdcurr, sharpfact, nbase, mod_cats,
                                 can_mods_offsets, mod_cat_weights,
                                 mod_weight);
    }

    // Final score is sum of final state + its stay
    float score = fwd[nblk * nseqpos + nseqpos - 1];
    return score;
}


void cm_flipflop_backward_step(float const * logprob, float const * bwdprev,
                               int32_t const * seq, size_t nseqpos,
                               float * bwdcurr, float sharpfact,
                               size_t nbase, int32_t const * mod_cats,
                               int32_t const * can_mods_offsets,
                               float const * mod_cat_weights,
                               float mod_weight){
    assert(nseqpos > 0);
    assert(NULL != logprob);
    assert(NULL != bwdprev);
    assert(NULL != seq);
    assert(NULL != bwdcurr);
    assert(NULL != mod_cats);
    assert(NULL != can_mods_offsets);

    assert(nbase > 0);
    const size_t offset_stay = nbase * (nbase + nbase);
    const size_t offset_mod = offset_stay + nbase + nbase;

    for(size_t pos=0 ; pos < nseqpos ; pos++){
        // Stay in current position
        const size_t base = seq[pos];
        bwdcurr[pos] = (base < nbase) ? logprob[base * (nbase + nbase) + base]:
            logprob[offset_stay + base];
        bwdcurr[pos] += bwdprev[pos];
    }
    for(size_t pos=1 ; pos < nseqpos ; pos++){
        // Move to new position
        const size_t base_to = seq[pos];
        const size_t base_from = seq[pos - 1];
        size_t can_base_to = base_to;
        if(can_base_to >= nbase){
            can_base_to -= nbase;
        }

        assert(base_to != base_from);  // Can't have repeated bases
        assert(base_to < nbase || base_from + nbase == base_to);
        const float can_score = (base_to < nbase) ?
            logprob[base_to * (nbase + nbase) + base_from] :
            logprob[offset_stay + base_from];
        const size_t mod_idx = can_mods_offsets[can_base_to] + mod_cats[pos];
        const float mod_score =
            mod_weight * mod_cat_weights[mod_idx] *
            logprob[offset_mod + mod_idx];
        bwdcurr[pos - 1] = logsumexpf(bwdcurr[pos - 1],
                                      bwdprev[pos] + can_score + mod_score,
                                      sharpfact);
    }
}


float cm_flipflop_backward(float const * logprob, size_t nblk, size_t ldp,
                           int32_t const * seq, size_t nseqpos,
                           float sharpfact, float * bwd, size_t nbase,
                           int32_t const * mod_cats,
                           int32_t const * can_mods_offsets,
                           float const * mod_cat_weights,
                           float mod_weight){
    assert(nseqpos > 0);
    assert(NULL != logprob);
    assert(NULL != seq);
    assert(NULL != bwd);

    //  Point prior -- must have ended in either final stay or state
    for(size_t pos=0 ; pos < nseqpos ; pos++){
        bwd[nblk * nseqpos + pos] = -LARGE_VAL;
    }
    // Final stay
    bwd[nblk * nseqpos + nseqpos - 1] = 0.0;

    for(size_t blk=nblk ; blk > 0 ; blk--){
        float const * bwdprev = bwd + blk * nseqpos;
        float * bwdcurr = bwd + (blk - 1) * nseqpos;
        float const * logprobcurr = logprob + (blk - 1) * ldp;

        cm_flipflop_backward_step(logprobcurr, bwdprev, seq, nseqpos,
                                  bwdcurr, sharpfact, nbase, mod_cats,
                                  can_mods_offsets, mod_cat_weights,
                                  mod_weight);
    }

    return bwd[0];
}


void cat_mod_flipflop_cost(float const * logprob, size_t nstate, size_t nblk,
                           size_t nbatch, int32_t const * seqs,
                           int32_t const * seqlen, int32_t const * mod_cats,
                           int32_t const * can_mods_offsets,
                           float const * mod_cat_weights, float mod_weight,
                           float sharpfact, float * score){
    // assumes 4 canonical bases (but confirms via conversion)
    const size_t nbase = nstate_to_nbase(nstate - can_mods_offsets[4]);
    const size_t ldp = nbatch * nstate;
    size_t seqidx[nbatch];
    seqidx[0] = 0;
    for(size_t idx=1 ; idx < nbatch ; idx++){
        seqidx[idx] = seqidx[idx - 1] + seqlen[idx - 1];
    }

#pragma omp parallel for
    for(size_t batch=0 ; batch < nbatch ; batch++){
         if(0 == seqlen[batch]){
             score[batch] = 0.0;
             continue;
         }

        const size_t batch_offset = batch * nstate;
        float * fwd = calloc((1 + nblk) * seqlen[batch], sizeof(float));
        score[batch] =
            cm_flipflop_forward(logprob + batch_offset, nblk, ldp,
                                seqs + seqidx[batch], seqlen[batch],
                                sharpfact, fwd, nbase,
                                mod_cats + seqidx[batch],
                                can_mods_offsets, mod_cat_weights,
                                mod_weight);
        free(fwd);
    }
}


void cm_flipflop_scores_fwd(float const * logprob, size_t nstate, size_t nblk,
                            size_t nbatch, int32_t const * seqs,
                            int32_t const * seqlen, int32_t const * mod_cats,
                            int32_t const * can_mods_offsets,
                            float const * mod_cat_weights, float mod_weight,
                            float sharpfact, float * score){
    cat_mod_flipflop_cost(logprob, nstate, nblk, nbatch, seqs, seqlen, mod_cats,
                          can_mods_offsets, mod_cat_weights, mod_weight,
                          sharpfact, score);
}


void cm_flipflop_scores_bwd(float const * logprob, size_t nstate,
                            size_t nblk, size_t nbatch, int32_t const * seqs,
                            int32_t const * seqlen, int32_t const * mod_cats,
                            int32_t const * can_mods_offsets,
                            float const * mod_cat_weights, float mod_weight,
                            float sharpfact, float * score){
    size_t nbase = nstate_to_nbase(nstate - can_mods_offsets[4]);
    size_t ldp = nbatch * nstate;
    size_t seqidx[nbatch];
    seqidx[0] = 0;
    for(size_t idx=1 ; idx < nbatch ; idx++){
        seqidx[idx] = seqidx[idx - 1] + seqlen[idx - 1];
    }

#pragma omp parallel for
    for(size_t batch=0 ; batch < nbatch ; batch++){
         if(0 == seqlen[batch]){
             score[batch] = 0.0;
             continue;
         }
        const size_t offset = batch * nstate;
        float * bwd = calloc((1 + nblk) * seqlen[batch], sizeof(float));
        score[batch] =
          cm_flipflop_backward(logprob + offset, nblk, ldp,
                               seqs + seqidx[batch], seqlen[batch],
                               sharpfact, bwd, nbase,
                               mod_cats + seqidx[batch], can_mods_offsets,
                               mod_cat_weights, mod_weight);
        free(bwd);
    }
}


void cm_flipflop_grad_step(float const * fwdcurr, float const * bwdnext,
                           float const * logprob, int32_t const * seq,
                           int32_t nseqpos, int32_t const * mod_cats,
                           int32_t const * can_mods_offsets,
                           float const * mod_cat_weights, float mod_weight,
                           float * grad, size_t nstate,
                           float fact, float sharpfact){
    const size_t nbase = nstate_to_nbase(nstate - can_mods_offsets[4]);
    const size_t offset_stay = nbase * (nbase + nbase);
    const size_t offset_mod = offset_stay + nbase + nbase;

    // Make sure gradient calc is zero'd
    memset(grad, 0, nstate * sizeof(float));

    for(size_t pos=0 ; pos < nseqpos ; pos++){
        // Stay state
        const size_t base = seq[pos];
        const size_t idx = (base < nbase) ?
          (base * (nbase + nbase) + base) :
          (offset_stay + base);
        grad[idx] += expf(sharpfact * (fwdcurr[pos] + bwdnext[pos] +
                                       logprob[idx] - fact));
    }
    for(size_t pos=1 ; pos < nseqpos ; pos++){
        // Move state
        const size_t base_to = seq[pos];
        const size_t base_from = seq[pos - 1];
        size_t can_base_to = base_to;
        if(can_base_to >= nbase){
            can_base_to -= nbase;
        }
        const size_t can_idx = (base_to < nbase) ?
            (base_to * (nbase + nbase) + base_from) :
            (offset_stay + base_from);
        const size_t mod_idx = can_mods_offsets[can_base_to] + mod_cats[pos];

        assert(base_to != base_from);  // Can't have repeated bases
        assert(base_to < nbase || base_from + nbase == base_to);
        grad[can_idx] +=
          expf(sharpfact * (fwdcurr[pos - 1] + bwdnext[pos] +
                            logprob[can_idx] +
                            mod_weight * mod_cat_weights[mod_idx] *
                            logprob[offset_mod + mod_idx] - fact));
        grad[offset_mod + mod_idx] +=
          expf(sharpfact * (fwdcurr[pos - 1] + bwdnext[pos] +
                            logprob[can_idx] +
                            mod_weight * mod_cat_weights[mod_idx] *
                            logprob[offset_mod + mod_idx] - fact)) *
            mod_weight * mod_cat_weights[mod_idx];
    }
}


void cat_mod_flipflop_grad(float const * logprob, size_t nstate, size_t nblk,
                           size_t nbatch, int32_t const * seqs,
                           int32_t const * seqlen, int32_t const * mod_cats,
                           int32_t const * can_mods_offsets,
                           float const * mod_cat_weights, float mod_weight,
                           float sharpfact, float * grad){
    const size_t ldp = nbatch * nstate;
    const size_t nbase = nstate_to_nbase(nstate - can_mods_offsets[4]);
    size_t seqidx[nbatch];
    seqidx[0] = 0;
    for(size_t idx=1 ; idx < nbatch ; idx++){
        seqidx[idx] = seqidx[idx - 1] + seqlen[idx - 1];
    }

#pragma omp parallel for
    for(size_t batch=0 ; batch < nbatch ; batch++){
        const size_t batch_offset = batch * nstate;
        if(0 == seqlen[batch]){
            for(size_t blk=0 ; blk < nblk ; blk++){
                memset(grad + batch_offset + blk * ldp, 0,
                       nstate * sizeof(float));
            }
            continue;
        }
        const int32_t nseqpos = seqlen[batch];
        int32_t const * seq = seqs + seqidx[batch];
        int32_t const * mod_cat = mod_cats + seqidx[batch];
        float * fwd = calloc((nblk + 1) * nseqpos, sizeof(float));
        float * bwd = calloc((nblk + 1) * nseqpos, sizeof(float));
        cm_flipflop_forward(logprob + batch_offset, nblk, ldp, seq,
                            nseqpos, sharpfact, fwd, nbase, mod_cat,
                            can_mods_offsets, mod_cat_weights, mod_weight);
        cm_flipflop_backward(logprob + batch_offset, nblk, ldp, seq,
                             nseqpos, sharpfact, bwd, nbase, mod_cat,
                             can_mods_offsets, mod_cat_weights, mod_weight);

        // Normalised transition matrix
        for(size_t blk=0 ; blk < nblk ; blk++){
            float const * fwdcurr = fwd + blk * nseqpos;
            float const * bwdcurr = bwd + blk * nseqpos;
            float const * bwdnext = bwd + blk * nseqpos + nseqpos;
            float const * logprobcurr = logprob + batch_offset + blk * ldp;
            float * gradcurr = grad + batch_offset + blk * ldp;

            //  Recalculate close to position to reduce numerical error
            float fact = fwdcurr[0] + bwdcurr[0];
            for(size_t pos=1; pos < nseqpos ; pos++){
                fact = logsumexpf(fact, fwdcurr[pos] + bwdcurr[pos], sharpfact);
            }

            cm_flipflop_grad_step(fwdcurr, bwdnext, logprobcurr, seq, nseqpos,
                                  mod_cat, can_mods_offsets, mod_cat_weights,
                                  mod_weight, gradcurr, nstate, fact,
                                  sharpfact);
        }

        free(bwd);
        free(fwd);
    }
}


#ifdef CAT_MOD_FLIPFLOP_TEST

// flip-flip states : 0     0     1    5    5    1    3    2
// transition states:    0     8    33   37   13   25   19
// mod transitions  :   NA     1     0   NA    1    0    0
const int32_t test_seq1[12] = {0, 1, 5, 1, 3, 2,
                               0, 1, 5, 1, 3, 2};

const int32_t test_mod1[12] = {0, 0, 1, 1, 0, 0,
                               0, 1, 0, 1, 0, 0};

// mods only for C base (second flip-flop state)
const float test_mod_weight = 4.0;
const int32_t test_can_mod_offsets[5] = {0, 1, 3, 4, 5};
const float test_mod_cat_weights[5] = {1.0, 1.0, 80.0, 1.0, 1.0};

const int32_t test_seqlen1[2] = {6, 6};

float test_logprob1[630] = {
    // t = 0, blk = 0 -- stay in 0
    0.7137395145, 0.0058640570, 0.0043273252, 0.0057024065, 0.0001304555, 0.0167860687, 0.0014591201, 0.0039324691,
    0.0117071924, 0.0045297625, 0.0105104226, 0.0018303745, 0.0004133878, 0.0121020079, 0.0179132788, 0.0008446391,
    0.0003954364, 0.0046109826, 0.0061280611, 0.0037487558, 0.0002867797, 0.0021094619, 0.0090478168, 0.0088021810,
    0.0166425156, 0.0008985700, 0.0030807985, 0.0150129722, 0.0033072104, 0.0225965258, 0.0017120223, 0.0080003635,
    0.0086164755, 0.0085638228, 0.0090326148, 0.0184277679, 0.0128914220, 0.0024000880, 0.0143853339, 0.0075095396,
    1.0, 0.9, 0.1, 1.0, 1.0,
    // t = 0, blk = 1
    0.7137395145, 0.0058640570, 0.0043273252, 0.0057024065, 0.0001304555, 0.0167860687, 0.0014591201, 0.0039324691,
    0.0117071924, 0.0045297625, 0.0105104226, 0.0018303745, 0.0004133878, 0.0121020079, 0.0179132788, 0.0008446391,
    0.0003954364, 0.0046109826, 0.0061280611, 0.0037487558, 0.0002867797, 0.0021094619, 0.0090478168, 0.0088021810,
    0.0166425156, 0.0008985700, 0.0030807985, 0.0150129722, 0.0033072104, 0.0225965258, 0.0017120223, 0.0080003635,
    0.0086164755, 0.0085638228, 0.0090326148, 0.0184277679, 0.0128914220, 0.0024000880, 0.0143853339, 0.0075095396,
    1.0, 0.9, 0.1, 1.0, 1.0,

    // t = 1, blk = 0 -- move 0 to 1
    0.0138651518, 0.0068715546, 0.0137762669, 0.0142378858, 0.0038887475, 0.0002837213, 0.0009213002, 0.0046096374,
    0.7005158726, 0.0041189393, 0.0057012358, 0.0196555714, 0.0034917922, 0.0031160895, 0.0027309383, 0.0068903076,
    0.0016565445, 0.0013069584, 0.0067694923, 0.0071836470, 0.0012639324, 0.0110877851, 0.0064367276, 0.0085079412,
    0.0003521574, 0.0035635810, 0.0043749238, 0.0027222466, 0.0139259729, 0.0152291942, 0.0044505049, 0.0039157630,
    0.0096219943, 0.0208794052, 0.0031593320, 0.0516253381, 0.0051720879, 0.0038762056, 0.0067444477, 0.0014988048,
    1.0, 0.9, 0.1, 1.0, 1.0,
    // t = 1, blk = 1
    0.0138651518, 0.0068715546, 0.0137762669, 0.0142378858, 0.0038887475, 0.0002837213, 0.0009213002, 0.0046096374,
    0.7005158726, 0.0041189393, 0.0057012358, 0.0196555714, 0.0034917922, 0.0031160895, 0.0027309383, 0.0068903076,
    0.0016565445, 0.0013069584, 0.0067694923, 0.0071836470, 0.0012639324, 0.0110877851, 0.0064367276, 0.0085079412,
    0.0003521574, 0.0035635810, 0.0043749238, 0.0027222466, 0.0139259729, 0.0152291942, 0.0044505049, 0.0039157630,
    0.0096219943, 0.0208794052, 0.0031593320, 0.0516253381, 0.0051720879, 0.0038762056, 0.0067444477, 0.0014988048,
    1.0, 0.1, 0.9, 1.0, 1.0,

    // t = 2, blk = 0 -- move 1 to 5
    0.0104973116, 0.0278749046, 0.0016333734, 0.0132478834, 0.0108985734, 0.0326813004, 0.0104401808, 0.0281931252,
    0.0002602418, 0.0004849826, 0.0069461090, 0.0337142774, 0.0066522165, 0.0002687968, 0.0081917502, 0.0014596191,
    0.0033038509, 0.0071742025, 0.0079209436, 0.0027446117, 0.0001922884, 0.0002173728, 0.0022822792, 0.0063767010,
    0.0062269709, 0.0008360773, 0.0009815072, 0.0138239322, 0.0006819603, 0.0004184386, 0.0005169712, 0.0038701156,
    0.0018582183, 0.7184016070, 0.0038719050, 0.0057834926, 0.0016248741, 0.0121355831, 0.0023164603, 0.0029949899,
    1.0, 0.1, 0.9, 1.0, 1.0,
    // t = 2, blk = 1
    0.0104973116, 0.0278749046, 0.0016333734, 0.0132478834, 0.0108985734, 0.0326813004, 0.0104401808, 0.0281931252,
    0.0002602418, 0.0004849826, 0.0069461090, 0.0337142774, 0.0066522165, 0.0002687968, 0.0081917502, 0.0014596191,
    0.0033038509, 0.0071742025, 0.0079209436, 0.0027446117, 0.0001922884, 0.0002173728, 0.0022822792, 0.0063767010,
    0.0062269709, 0.0008360773, 0.0009815072, 0.0138239322, 0.0006819603, 0.0004184386, 0.0005169712, 0.0038701156,
    0.0018582183, 0.7184016070, 0.0038719050, 0.0057834926, 0.0016248741, 0.0121355831, 0.0023164603, 0.0029949899,
    1.0, 0.9, 0.1, 1.0, 1.0,

    // t = 3, blk = 0 -- stay in 5
    0.0132238486, 0.0067462421, 0.0065735995, 0.0002313058, 0.0350482900, 0.0038167453, 0.0013436872, 0.0047910351,
    0.0005511208, 0.0152455357, 0.0002505248, 0.0009566527, 0.0016608534, 0.0036526310, 0.0038930839, 0.0102019269,
    0.0040538124, 0.0121608248, 0.0026858640, 0.0024698387, 0.0077258147, 0.0063036375, 0.0015254714, 0.0015248249,
    0.0008483379, 0.0194108435, 0.0065140833, 0.0189690442, 0.0005446999, 0.0072716624, 0.0002782992, 0.0124768655,
    0.0239038132, 0.0108786276, 0.0208670656, 0.0076679875, 0.0086667116, 0.7072362974, 0.0038886950, 0.0039397951,
    1.0, 0.9, 0.1, 1.0, 1.0,
    // t = 3, blk = 1
    0.0132238486, 0.0067462421, 0.0065735995, 0.0002313058, 0.0350482900, 0.0038167453, 0.0013436872, 0.0047910351,
    0.0005511208, 0.0152455357, 0.0002505248, 0.0009566527, 0.0016608534, 0.0036526310, 0.0038930839, 0.0102019269,
    0.0040538124, 0.0121608248, 0.0026858640, 0.0024698387, 0.0077258147, 0.0063036375, 0.0015254714, 0.0015248249,
    0.0008483379, 0.0194108435, 0.0065140833, 0.0189690442, 0.0005446999, 0.0072716624, 0.0002782992, 0.0124768655,
    0.0239038132, 0.0108786276, 0.0208670656, 0.0076679875, 0.0086667116, 0.7072362974, 0.0038886950, 0.0039397951,
    1.0, 0.9, 0.1, 1.0, 1.0,

    // t = 4, blk = 0 -- move 5 to 1
    0.0162499295, 0.0042696969, 0.0190051755, 0.0162959320, 0.0038385851, 0.0010900080, 0.0051636429, 0.0088802400,
    0.0035193397, 0.0100004109, 0.0182444400, 0.0002015949, 0.0051056114, 0.7237303612, 0.0135142243, 0.0065390854,
    0.0029951279, 0.0029123437, 0.0010848643, 0.0320041842, 0.0029855054, 0.0001557548, 0.0043323211, 0.0161734933,
    0.0051668898, 0.0007899601, 0.0024293827, 0.0107437912, 0.0005963283, 0.0004204642, 0.0008271684, 0.0036831630,
    0.0058302092, 0.0044612666, 0.0090699795, 0.0135366090, 0.0087714458, 0.0033968323, 0.0002088134, 0.0117758241,
    1.0, 0.1, 0.9, 1.0, 1.0,
    // t = 4, blk = 1
    0.0162499295, 0.0042696969, 0.0190051755, 0.0162959320, 0.0038385851, 0.0010900080, 0.0051636429, 0.0088802400,
    0.0035193397, 0.0100004109, 0.0182444400, 0.0002015949, 0.0051056114, 0.7237303612, 0.0135142243, 0.0065390854,
    0.0029951279, 0.0029123437, 0.0010848643, 0.0320041842, 0.0029855054, 0.0001557548, 0.0043323211, 0.0161734933,
    0.0051668898, 0.0007899601, 0.0024293827, 0.0107437912, 0.0005963283, 0.0004204642, 0.0008271684, 0.0036831630,
    0.0058302092, 0.0044612666, 0.0090699795, 0.0135366090, 0.0087714458, 0.0033968323, 0.0002088134, 0.0117758241,
    1.0, 0.1, 0.9, 1.0, 1.0,

    // t = 5, blk = 0 -- move 1 to 3
    0.0054995373, 0.0003135968, 0.0036685129, 0.0239510419, 0.0039243790, 0.0019827996, 0.0129521071, 0.0066243852,
    0.0072536818, 0.0159209645, 0.0116239255, 0.0211135167, 0.0071678950, 0.0168522449, 0.0034948831, 0.0148879133,
    0.0084620257, 0.0075577618, 0.0042788046, 0.0007793942, 0.0038023124, 0.0116145280, 0.0025982395, 0.0022352670,
    0.0019744321, 0.7117781744, 0.0044554214, 0.0010030397, 0.0047838417, 0.0005540779, 0.0085588124, 0.0001078087,
    0.0019562465, 0.0097635189, 0.0012854310, 0.0076597643, 0.0032004197, 0.0354927128, 0.0017610103, 0.0071055704,
    1.0, 0.9, 0.1, 1.0, 1.0,
    // t = 5, blk = 1
    0.0054995373, 0.0003135968, 0.0036685129, 0.0239510419, 0.0039243790, 0.0019827996, 0.0129521071, 0.0066243852,
    0.0072536818, 0.0159209645, 0.0116239255, 0.0211135167, 0.0071678950, 0.0168522449, 0.0034948831, 0.0148879133,
    0.0084620257, 0.0075577618, 0.0042788046, 0.0007793942, 0.0038023124, 0.0116145280, 0.0025982395, 0.0022352670,
    0.0019744321, 0.7117781744, 0.0044554214, 0.0010030397, 0.0047838417, 0.0005540779, 0.0085588124, 0.0001078087,
    0.0019562465, 0.0097635189, 0.0012854310, 0.0076597643, 0.0032004197, 0.0354927128, 0.0017610103, 0.0071055704,
    1.0, 0.9, 0.1, 1.0, 1.0,

    // t = 6, blk = 0 -- move 3 to 2
    0.0027753489, 0.0042800652, 0.0131082339, 0.0027542745, 0.0073969560, 0.0022332778, 0.0063905429, 0.0225312653,
    0.0083716146, 0.0018647020, 0.0080511935, 0.0062377027, 0.0096483698, 0.0050934491, 0.0002518356, 0.0089501860,
    0.0019424988, 0.0028867039, 0.0362414220, 0.7084635261, 0.0012042079, 0.0016243873, 0.0089677837, 0.0001407093,
    0.0007788545, 0.0061531496, 0.0116723082, 0.0160689361, 0.0045947877, 0.0025051798, 0.0016243552, 0.0025087153,
    0.0037103848, 0.0021407879, 0.0141961964, 0.0206362499, 0.0234809816, 0.0151728742, 0.0018537195, 0.0014922626,
    1.0, 0.9, 0.1, 1.0, 1.0,
    // t = 6, blk = 1
    0.0027753489, 0.0042800652, 0.0131082339, 0.0027542745, 0.0073969560, 0.0022332778, 0.0063905429, 0.0225312653,
    0.0083716146, 0.0018647020, 0.0080511935, 0.0062377027, 0.0096483698, 0.0050934491, 0.0002518356, 0.0089501860,
    0.0019424988, 0.0028867039, 0.0362414220, 0.7084635261, 0.0012042079, 0.0016243873, 0.0089677837, 0.0001407093,
    0.0007788545, 0.0061531496, 0.0116723082, 0.0160689361, 0.0045947877, 0.0025051798, 0.0016243552, 0.0025087153,
    0.0037103848, 0.0021407879, 0.0141961964, 0.0206362499, 0.0234809816, 0.0151728742, 0.0018537195, 0.0014922626,
    1.0, 0.9, 0.1, 1.0, 1.0,
};



#include <stdio.h>

int main(int argc, char * argv[]){

    const size_t nblk = 7;
    const size_t nstate = 45;
    const size_t nbatch = 2;
    float score[2] = {0.0f};
    const float DELTA = 1e-2f;
    const float sharpfact = (argc > 1) ? atof(argv[1]) : 1.0f;
    const size_t msize = nblk * nstate * nbatch;

    for(size_t i=0 ; i < msize ; i++){
        test_logprob1[i] = logf(test_logprob1[i]);
    }

    //
    //    F / B calculations
    //
    cm_flipflop_scores_fwd(test_logprob1, nstate, nblk, nbatch, test_seq1,
                           test_seqlen1, test_mod1, test_can_mod_offsets,
                           test_mod_cat_weights, test_mod_weight,
                           sharpfact, score);
    printf("Forwards scores: %f %f\n", score[0], score[1]);

    cm_flipflop_scores_bwd(test_logprob1, nstate, nblk, nbatch, test_seq1,
                            test_seqlen1, test_mod1, test_can_mod_offsets,
                            test_mod_cat_weights, test_mod_weight,
                            sharpfact, score);
    printf("Backwards scores: %f %f\n", score[0], score[1]);

    float * grad = calloc(msize, sizeof(float));
    cat_mod_flipflop_grad(test_logprob1, nstate, nblk, nbatch, test_seq1,
                          test_seqlen1, test_mod1, test_can_mod_offsets,
                          test_mod_cat_weights, test_mod_weight,
                          sharpfact, grad);
    float maxdelta = 0.0;
    for(size_t blk=0 ; blk < nblk ; blk++){
        const size_t offset = blk * nbatch * nstate;
        for(size_t st=0 ; st < nstate ; st++){
            maxdelta = fmaxf(maxdelta, fabsf(grad[offset + st] -
                                             grad[offset + nstate + st]));
        }
    }
    printf("Max grad delta = %f\n", maxdelta);

    printf("Derviatives:\n");
    float fscore[2] = {0.0f};
    for(size_t blk=0 ; blk < nblk ; blk++){
        printf("  Block %zu\n", blk);
        const size_t offset = blk * nbatch * nstate;
        for(size_t st=0 ; st < nstate ; st++){
            // Positive difference
            const float orig = test_logprob1[offset + st];
            test_logprob1[offset + st] = orig + DELTA;
            cm_flipflop_scores_fwd(test_logprob1, nstate, nblk, nbatch,
                                   test_seq1, test_seqlen1, test_mod1,
                                   test_can_mod_offsets, test_mod_cat_weights,
                                   test_mod_weight, sharpfact, score);
            fscore[0] = score[0];
            fscore[1] = score[1];
            // Negative difference
            test_logprob1[offset + st] = orig - DELTA;
            cm_flipflop_scores_fwd(test_logprob1, nstate, nblk, nbatch,
                                   test_seq1, test_seqlen1, test_mod1,
                                   test_can_mod_offsets, test_mod_cat_weights,
                                   test_mod_weight, sharpfact, score);
            fscore[0] = (fscore[0] - score[0]) / (2.0f * DELTA);
            fscore[1] = (fscore[1] - score[1]) / (2.0f * DELTA);
            // Report and reset
            test_logprob1[offset + st] = orig;
            printf("    %f d=%f r=%f [%f %f]\n", grad[offset + st],
                   fabsf(grad[offset + st] - fscore[0]),
                   grad[offset + st] / fscore[0], fscore[0], fscore[1]);
        }
    }
}

#endif /* CAT_MOD_FLIPFLOP_TEST */
