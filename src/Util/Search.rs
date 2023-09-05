use arrayfire;




use rayon::prelude::*;







pub fn COO_find(
	WRowIdxCOO: &arrayfire::Array<i32>,
    target_rows: &arrayfire::Array<i32>
    ) -> arrayfire::Array<i32>
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

    arrayfire::locate(&bool_result).cast::<i32>()
}











