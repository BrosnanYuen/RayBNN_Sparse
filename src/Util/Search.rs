use arrayfire;




use rayon::prelude::*;







pub fn COO_find<Z: arrayfire::RealNumber>(
	WRowIdxCOO: &arrayfire::Array<Z>,
    target_rows: &arrayfire::Array<Z>
    ) -> arrayfire::Array<Z>
{
    let target_row_num  = target_rows.dims()[0];
    let WRowIdxCOO_num  = WRowIdxCOO.dims()[0];
    let COO_dims = arrayfire::Dim4::new(&[1,target_row_num,1,1]);
    let WRowIdxCOO_tile = arrayfire::tile(WRowIdxCOO, COO_dims);
    let mut trans_rows = arrayfire::transpose(target_rows,false);
    let trans_rows_dims = arrayfire::Dim4::new(&[WRowIdxCOO_num,1,1,1]);
    trans_rows = arrayfire::tile(&trans_rows, trans_rows_dims);

    let mut bool_result = arrayfire::eq(&WRowIdxCOO_tile, &trans_rows, false);

    bool_result = arrayfire::any_true(&bool_result, 1);

    arrayfire::locate(&bool_result).cast::<Z>()
}








pub fn COO_batch_find<Z: arrayfire::RealNumber>(
	WRowIdxCOO: &arrayfire::Array<Z>,
    target_rows: &arrayfire::Array<Z>,
    batch_size: u64
    ) -> arrayfire::Array<Z>
{

    let mut i: u64 = 0;
    let mut startseq: u64 = 0;
    let mut endseq: u64 = 0;
    let total_size = target_rows.dims()[0];





    startseq = i;
    endseq = i + batch_size-1;
    if (endseq >= (total_size-1))
    {
        endseq = total_size-1;
    }

    let inputarr  = arrayfire::rows(target_rows, startseq  as i64,endseq as i64);

    let mut total_idx= COO_find(
        WRowIdxCOO,
        &inputarr
        );
    i = i + batch_size;


    while i < total_size
    {
        startseq = i;
        endseq = i + batch_size-1;
        if (endseq >= (total_size-1))
        {
            endseq = total_size-1;
        }

        let inputarr  = arrayfire::rows(target_rows, startseq  as i64,endseq as i64);

        let idx= COO_find(
            WRowIdxCOO,
            &inputarr
            );
        
        if (idx.dims()[0] > 0)
        {
            total_idx = arrayfire::join(0, &total_idx, &idx);
        }

        i = i + batch_size;
    }



    arrayfire::sort(&total_idx,0,true)
}









pub fn find_unique<Z: arrayfire::IndexableType>(
    arr: &arrayfire::Array<Z>,
    neuron_size: u64
    ) -> arrayfire::Array<Z>
    {

    let table_dims = arrayfire::Dim4::new(&[neuron_size,1,1,1]);
    let mut table = arrayfire::constant::<bool>(false,table_dims);

    let inarr = arrayfire::constant::<bool>(true, arr.dims());
    //let idxarr = arr.cast::<u32>();

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(arr, 0, None);
    arrayfire::assign_gen(&mut table, &idxrs, &inarr);

    arrayfire::locate(&table).cast::<Z>()
}






pub fn parallel_lookup<Z: arrayfire::HasAfEnum, V: arrayfire::IndexableType>(
	batch_dim: u64,
	lookup_dim: u64,

	idx: &arrayfire::Array<V>,
	target: &arrayfire::Array<Z>,
) ->  arrayfire::Array<Z>
{

	let target_dims = target.dims();

	let batch_num = target_dims[batch_dim as usize];
	let lookup_size = target_dims[lookup_dim as usize];

	
	let repeat_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let mut tile_dims = arrayfire::Dim4::new(&[1,1,1,1]);

	tile_dims[batch_dim as usize] = batch_num;

	let count =  arrayfire::iota::<V>(tile_dims,repeat_dims);

    let batch_num_V = arrayfire::constant::<u64>(batch_num,repeat_dims).cast::<V>();

	let mut idx2 = batch_num_V*idx.clone();
	
	idx2 = arrayfire::add(&idx2, &count, true);

	drop(count);





	

	idx2 = arrayfire::flat(&idx2);

	let mut output_arr = arrayfire::flat(target);

	output_arr = arrayfire::lookup(&output_arr, &idx2, 0);


	drop(idx2);
	
	arrayfire::moddims(&output_arr, idx.dims())
}








pub fn integer_histogram(
	input: &arrayfire::Array<i32>,

    bins: &mut arrayfire::Array<i32>,
    counts: &mut arrayfire::Array<u32>
    ) {


    let sorted = arrayfire::sort(&input, 0, true);


    let ones = arrayfire::constant::<i32>(1,sorted.dims());
    let  (keys, values) = arrayfire::sum_by_key(&sorted, &ones, 0);

    *bins = keys;
    *counts = values.cast::<u32>();
}




