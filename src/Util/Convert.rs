use arrayfire;




use rayon::prelude::*;



pub fn COO_to_CSR<Z: arrayfire::IndexableType + arrayfire::ReduceByKeyInput>(
	WRowIdxCOO: &arrayfire::Array<Z>,
    row_num: u64
    ) -> arrayfire::Array<Z>
	{

    let WRowIdxCOO_num  = WRowIdxCOO.dims()[0];


    let ones = arrayfire::constant::<u64>(1,arrayfire::Dim4::new(&[WRowIdxCOO_num,1,1,1]));
    let ones = ones.cast::<Z>();
    let mut temparr = arrayfire::constant::<u64>(0,arrayfire::Dim4::new(&[row_num,1,1,1]));
    let mut temparr = temparr.cast::<Z>();

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(WRowIdxCOO, 0, None);
    arrayfire::assign_gen(&mut temparr, &idxrs, &ones);

    let sel = arrayfire::locate(&temparr);


    //let  (_,mut sumarr) = arrayfire::count_by_key(WRowIdxCOO, &ones, 0);
    let  (_,mut sumarr) = arrayfire::sum_by_key(WRowIdxCOO, &ones, 0);

    let sumarr = sumarr.cast::<Z>();


    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&sel, 0, None);
    arrayfire::assign_gen(&mut temparr, &idxrs, &sumarr);



    temparr = arrayfire::accum(&temparr, 0).cast::<Z>();


    let constarr = arrayfire::constant::<u64>(0,arrayfire::Dim4::new(&[1,1,1,1]));
    let constarr = constarr.cast::<Z>();
    temparr = arrayfire::join(0, &constarr, &temparr);

    temparr
}











fn gen_const(pair: (usize, i32)) -> Vec<i32>
{
    let (i,e) = pair;
    let a: Vec<i32> = vec![i as i32; e as usize];
    a
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


