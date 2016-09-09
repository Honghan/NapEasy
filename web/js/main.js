(function($){
	var files = ["Ahn et al., (2011) - The cortical neuroanatomy of neuropsychological deficits in MCI and AD_annotated_ann.sum","Allert et al., (2011) - Role of dysphagia in evaluating PD for STN-DBS._annotated_ann.sum","Altug et al., (2011) - The influence of subthalamic nucleus DBS on daily living activities in PD_annotated_ann.sum","Altug et al., (2014) - Brain stimulation of subthalamic nucleus surgery on gait and balance in PD_annotated_ann.sum","Arnaldi et al., (2016) - Functional neuroimaging and clinical features of drug naive patients with de novo PD and RBD_annotated_ann.sum","Avila et al., (2011) - Effect of temporal lobe structure volume on memory in eldery depressed patients_annotated_ann.sum","Bartova et al., (2010) - Correlation between substantia nigra features detected by sonography and PD_annotated_ann.sum","Benninger et al., (2008) - Morphological differences in PD with and without rest tremor_annotated_ann.sum"];
	$(document).ready(function(){
		for(var i=0;i<files.length;i++){
			$('#file_list').append($("<option></option>")
	                    .attr("value",files[i])
	                    .text(decodeURIComponent(files[i]))); 
		}

		$('#file_list').on('change', function() {
			tmining.load_text(this.value);
		});

		tmining.load_text($("#file_list option:selected").text());
	})

})(this.jQuery)