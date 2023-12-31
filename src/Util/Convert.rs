use arrayfire;




use rayon::prelude::*;


pub fn get_global_weight_idx(
    neuron_size: u64,
    WRowIdxCOO: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,
) -> arrayfire::Array<u64>
{


    (WRowIdxCOO.cast::<u64>()*(neuron_size)) +  WColIdx.cast::<u64>()
}



pub fn get_global_weight_idx2(
    neuron_size: u64,
    WRowIdxCOO: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,
) -> arrayfire::Array<u64>
{


    WRowIdxCOO.cast::<u64>() +  (WColIdx.cast::<u64>()*(neuron_size))
}




fn gen_const(pair: (usize, i32)) -> Vec<i32>
{
    let (i,e) = pair;
    let a: Vec<i32> = vec![i as i32; e as usize];
    a
}










pub fn COO_to_CSR(
	WRowIdxCOO: &arrayfire::Array<i32>,
    row_num: u64
    ) -> arrayfire::Array<i32>
	{

    let WRowIdxCOO_num  = WRowIdxCOO.dims()[0];


    let ones = arrayfire::constant::<i32>(1,arrayfire::Dim4::new(&[WRowIdxCOO_num,1,1,1]));
    let mut temparr = arrayfire::constant::<i32>(0,arrayfire::Dim4::new(&[row_num,1,1,1]));

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(WRowIdxCOO, 0, None);
    arrayfire::assign_gen(&mut temparr, &idxrs, &ones);

    let sel = arrayfire::locate(&temparr);


    //let  (_,mut sumarr) = arrayfire::count_by_key(WRowIdxCOO, &ones, 0);
    let  (_,mut sumarr) = arrayfire::sum_by_key(WRowIdxCOO, &ones, 0);

    sumarr = sumarr.cast::<i32>();


    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&sel, 0, None);
    arrayfire::assign_gen(&mut temparr, &idxrs, &sumarr);



    temparr = arrayfire::accum(&temparr, 0);


    let constarr = arrayfire::constant::<i32>(0,arrayfire::Dim4::new(&[1,1,1,1]));
    temparr = arrayfire::join(0, &constarr, &temparr);

    temparr
}






pub fn CSR_to_COO(
	WRowIdxCSR: &arrayfire::Array<i32>
    ) -> arrayfire::Array<i32>
	{

	let rsize: u64 = WRowIdxCSR.dims()[0];
	let r0 = arrayfire::rows(WRowIdxCSR, 0, (rsize-2) as i64);
	let r1 = arrayfire::rows(WRowIdxCSR, 1, (rsize-1) as i64);

	let rowdiff = r1.clone() - r0.clone();


    let mut rowdiff_cpu = vec!(i32::default();rowdiff.elements());
	rowdiff.host(&mut rowdiff_cpu);



	let (count,_) = arrayfire::sum_all::<i32>(&rowdiff);

	let WRowIdxCOO_dims = arrayfire::Dim4::new(&[count as u64,1,1,1]);


	let mut WRowIdxCOO_cpu: Vec<i32> = rowdiff_cpu.into_par_iter().enumerate().map(gen_const).flatten_iter().collect();

    arrayfire::Array::new(&WRowIdxCOO_cpu, WRowIdxCOO_dims)
}











pub fn remap_rows(
	rowvec: &arrayfire::Array<i32>,
    idxsel: &arrayfire::Array<i32>,
    row_num: u64
    ) -> arrayfire::Array<i32>
{
    let table_dims = arrayfire::Dim4::new(&[row_num,1,1,1]);
    let mut table = arrayfire::constant::<i32>(0,table_dims);





    let single = arrayfire::Dim4::new(&[1,1,1,1]);
    
    let mut indexarr = arrayfire::iota::<i32>(idxsel.dims(),single);




    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(idxsel, 0, None);
    arrayfire::assign_gen(&mut table, &idxrs, &indexarr);



    arrayfire::lookup(&table, rowvec, 0)
}





