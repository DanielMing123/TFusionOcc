#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ void eval_bases24(
    float u, 
	float v, 
	float w, 
    float* __restrict__ Bx,
    float* __restrict__ By,
    float* __restrict__ Bz)
{
    float u2 = u*u, v2 = v*v, w2 = w*w;
    float r2 = u2 + v2;

    // Tier 1
    Bx[0]=1.f; By[0]=0.f; Bz[0]=0.f;
    Bx[1]=0.f; By[1]=1.f; Bz[1]=0.f;
    Bx[2]=0.f; By[2]=0.f; Bz[2]=1.f;

    Bx[3]=u;   By[3]=0.f; Bz[3]=0.f;
    Bx[4]=0.f; By[4]=v;   Bz[4]=0.f;
    Bx[5]=0.f; By[5]=0.f; Bz[5]=w;

    Bx[6]=v;   By[6]=0.f; Bz[6]=0.f;
    Bx[7]=w;   By[7]=0.f; Bz[7]=0.f;
    Bx[8]=0.f; By[8]=w;   Bz[8]=0.f;

    // Tier 2
    Bx[9]=0.f;  By[9]=u;   Bz[9]=0.f;
    Bx[10]=0.f; By[10]=0.f; Bz[10]=u;
    Bx[11]=0.f; By[11]=0.f; Bz[11]=v;

    Bx[12]= -w*v; By[12]=  w*u; Bz[12]=0.f;    // twist z
    Bx[13]= 0.f;  By[13]= -u*w; Bz[13]= u*v;   // twist x
    Bx[14]=  v*w; By[14]= 0.f;  Bz[14]= -v*u;  // twist y

    Bx[15]= w2; By[15]=0.f; Bz[15]=0.f;        // bend x by z^2
    Bx[16]=0.f; By[16]= w2; Bz[16]=0.f;        // bend y by z^2
    Bx[17]=0.f; By[17]=0.f; Bz[17]= r2;        // arch z vs r^2

    // Tier 3
    Bx[18]= u2; By[18]=0.f;  Bz[18]=0.f;
    Bx[19]=0.f; By[19]= v2;  Bz[19]=0.f;
    Bx[20]=0.f; By[20]=0.f;  Bz[20]= w2;

    Bx[21]= r2*u; By[21]= r2*v; Bz[21]=0.f;    // radial bulge
    Bx[22]= u*v;  By[22]= u*v;  Bz[22]=0.f;

    Bx[23]= u*v2; By[23]= v*u2; Bz[23]=0.f;    // softabs corner rounder
}

// Perform initial steps for each Gaussian prior to rasterization.
__global__ void preprocessCUDA(
	const int P,
	const int* points_xyz,
	const int* radii,
	const dim3 grid,
	uint32_t* tiles_touched)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	tiles_touched[idx] = 0;

	uint3 rect_min, rect_max;
	getRect(points_xyz + 3 * idx, radii[idx], rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) * (rect_max.z - rect_min.z) == 0)
		return;

	tiles_touched[idx] = (rect_max.z - rect_min.z) * (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void renderCUDA(
	const int N,
	const int scale_multiplier,
	const float* __restrict__ pts,
	const int* __restrict__ points_int,
	const dim3 grid,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const uint32_t* __restrict__ tiles_touched,
	const float* __restrict__ means3D,
	const float* __restrict__ scales3D,
	const float* __restrict__ rot3D,
	const float* __restrict__ opas,
	const float* __restrict__ u,
	const float* __restrict__ v,
	const float* __restrict__ semantic,
	const float* __restrict__ deform_weight,
	float* __restrict__ out_logits,
	float* __restrict__ out_bin_logits,
	float* __restrict__ out_density,
	float* __restrict__ out_probability)
{
    auto idx = cg::this_grid().thread_rank();
	if (idx >= N)
	    return;

	const int* point_int = points_int + idx * 3;
	const int voxel_idx = point_int[0] * grid.y * grid.z + point_int[1] * grid.z + point_int[2];
	const float3 point = {pts[3 * idx], pts[3 * idx + 1], pts[3 * idx + 2]};

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[voxel_idx];

	// Initialize helper variables
	float C[CHANNELS] = { 0 };
	float bin_logit = 1.0;
	float density = 0.0;
	float prob_sum = 0.0;

	for (int i = range.x; i < range.y; i++)
	{
		int gs_idx = point_list[i];
		float dof = max(cbrtf(tiles_touched[gs_idx]) / scale_multiplier, 1e-6f); // degree of freedom = total samples - 1, powf(cbrtf(tiles_touched[gs_idx]) / scale_multiplier, 3.0) - 1.0
		float3 rot1 = { rot3D[gs_idx * 9 + 0], rot3D[gs_idx * 9 + 1], rot3D[gs_idx * 9 + 2] };
		float3 rot2 = { rot3D[gs_idx * 9 + 3], rot3D[gs_idx * 9 + 4], rot3D[gs_idx * 9 + 5] };
		float3 rot3 = { rot3D[gs_idx * 9 + 6], rot3D[gs_idx * 9 + 7], rot3D[gs_idx * 9 + 8] }; 
		float3 d = { - means3D[gs_idx * 3] + point.x, - means3D[gs_idx * 3 + 1] + point.y, - means3D[gs_idx * 3 + 2] + point.z };
		float3 s = { scales3D[gs_idx * 3], scales3D[gs_idx * 3 + 1], scales3D[gs_idx * 3 + 2] };
		float3 ori_trans = { rot1.x * d.x + rot1.y * d.y + rot1.z * d.z, rot2.x * d.x + rot2.y * d.y + rot2.z * d.z, rot3.x * d.x + rot3.y * d.y + rot3.z * d.z }; //trans

		/////////// Introduce Inverse-Warp with 24 field-basis /////////
		float3 uvw = {ori_trans.x/s.x, ori_trans.y/s.y, ori_trans.z/s.z}; 
		float Bx[NUM_BASIS], By[NUM_BASIS], Bz[NUM_BASIS];
		eval_bases24(uvw.x, uvw.y, uvw.z, Bx, By, Bz);
		float3 correc_trans={0.f, 0.f, 0.f};
		for (int k=0; k<NUM_BASIS; k++){
			float wk = deform_weight[NUM_BASIS * gs_idx + k];
			correc_trans.x += wk * Bx[k];
			correc_trans.y += wk * By[k];
			correc_trans.z += wk * Bz[k];
		}
		float3 trans = {ori_trans.x - correc_trans.x, ori_trans.y - correc_trans.y, ori_trans.z - correc_trans.z};
		//////////////////////////////////////////////////////////////////

		float term_x = powf((trans.x / s.x) * (trans.x / s.x), 1.0f / u[gs_idx]); // u -> exponent2
		float term_y = powf((trans.y / s.y) * (trans.y / s.y), 1.0f / u[gs_idx]); 
		float term_z = powf((trans.z / s.z) * (trans.z / s.z), 1.0f / v[gs_idx]); // v -> exponent1
		float f = powf(term_x + term_y + 1e-6f, u[gs_idx] / v[gs_idx]) + term_z;
		float power = powf(1 + f / dof, -(dof+3)/2);
		// float logp = -0.5f * (dof + 3.0f) * log1pf(f / dof);
		// float power = expf(logp);
		float prob = power * opas[gs_idx];

		for (int ch = 0; ch < CHANNELS; ch++)
		{
			C[ch] += semantic[CHANNELS * gs_idx + ch] * prob;
		}
		bin_logit = (1 - power) * bin_logit;
		density = power + density;
		prob_sum = prob + prob_sum;
	}

	// Iterate over batches until all done or range is complete
	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (prob_sum > 1e-6f) {
		for (int ch = 0; ch < CHANNELS; ch++)
			out_logits[idx * CHANNELS + ch] = C[ch] / prob_sum;
	} else {
		for (int ch = 0; ch < CHANNELS; ch++) 
			out_logits[idx * CHANNELS + ch] = 1.0 / (CHANNELS); 
	}
	out_bin_logits[idx] = 1 - bin_logit;
	out_density[idx] = density;
	out_probability[idx] = prob_sum;
}


void FORWARD::render(
	const int N,
	const int scale_multiplier,
	const float* pts,
	const int* points_int,
	const dim3 grid,
	const uint2* ranges,
	const uint32_t* point_list,
	const uint32_t* tiles_touched,
	const float* means3D,
	const float* scales3D,
	const float* rot3D,
	const float* opas,
	const float* u,
	const float* v,
	const float* semantic,
	const float* deform_weight,
	float* out_logits,
	float* out_bin_logits,
	float* out_density,
	float* out_probability)
{
	renderCUDA<NUM_CHANNELS> << <(N + 255) / 256, 256 >> > (
		N, 
		scale_multiplier,
		pts,
		points_int,
		grid,
		ranges,
		point_list,
		tiles_touched,
		means3D,
		scales3D,
		rot3D,
		opas,
		u,
		v,
		semantic,
		deform_weight,
		out_logits,
		out_bin_logits,
		out_density,
		out_probability);
}


void FORWARD::preprocess(
	const int P,
	const int* points_xyz,
	const int* radii,
	const dim3 grid,
	uint32_t* tiles_touched)
{
	preprocessCUDA << <(P + 255) / 256, 256 >> > (
		P,
		points_xyz,
		radii,
		grid,
		tiles_touched
	);
}