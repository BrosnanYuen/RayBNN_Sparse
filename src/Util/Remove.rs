use arrayfire;




pub fn select_values<Z: arrayfire::FloatingPoint>(
    WValues: &mut arrayfire::Array<Z>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
    sel: &arrayfire::Array<u32>
)
{

    *WValues = arrayfire::lookup(WValues, sel, 0);
    *WRowIdxCOO = arrayfire::lookup(WRowIdxCOO, sel, 0);
    *WColIdx = arrayfire::lookup(WColIdx, sel, 0);

}




pub fn clear_input<Z: arrayfire::FloatingPoint>(
    WValues: &mut arrayfire::Array<Z>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
    input_rows: u64
)
{

    let single = input_rows as i32;

    //let cmp2 = (single <= WRowIdxCOO );
    let cmp1 = arrayfire::le(&single ,WRowIdxCOO, false);

    let sel = arrayfire::locate(&cmp1);

    select_values::<Z>(
        WValues,
        WRowIdxCOO,
        WColIdx,
        &sel
    );

}




