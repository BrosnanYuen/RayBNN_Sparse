use arrayfire;






pub fn COO_to_CSR<Z: arrayfire::ConstGenerator>(
	WRowIdxCOO: &arrayfire::Array<Z>,
    row_num: u64
    ) -> arrayfire::Array<Z>
	{

    let WRowIdxCOO_num  = WRowIdxCOO.dims()[0];


    let ones = arrayfire::constant::<Z>(1,arrayfire::Dim4::new(&[WRowIdxCOO_num,1,1,1]));
    let mut temparr = arrayfire::constant::<Z>(0,arrayfire::Dim4::new(&[row_num,1,1,1]));

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(WRowIdxCOO, 0, None);
    arrayfire::assign_gen(&mut temparr, &idxrs, &ones);

    let sel = arrayfire::locate(&temparr);


    //let  (_,mut sumarr) = arrayfire::count_by_key(WRowIdxCOO, &ones, 0);
    let  (_,mut sumarr) = arrayfire::sum_by_key(WRowIdxCOO, &ones, 0);

    sumarr = sumarr.cast::<Z>();


    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&sel, 0, None);
    arrayfire::assign_gen(&mut temparr, &idxrs, &sumarr);



    temparr = arrayfire::accum(&temparr, 0);


    let constarr = arrayfire::constant::<Z>(0,arrayfire::Dim4::new(&[1,1,1,1]));
    temparr = arrayfire::join(0, &constarr, &temparr);

    temparr
}



