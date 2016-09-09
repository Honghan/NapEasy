(function($){
	var files = ["Ahn et al., (2011) - The cortical neuroanatomy of neuropsychological deficits in MCI and AD_annotated_ann.sum","Allert et al., (2011) - Role of dysphagia in evaluating PD for STN-DBS._annotated_ann.sum","Altug et al., (2011) - The influence of subthalamic nucleus DBS on daily living activities in PD_annotated_ann.sum","Altug et al., (2014) - Brain stimulation of subthalamic nucleus surgery on gait and balance in PD_annotated_ann.sum","Arnaldi et al., (2016) - Functional neuroimaging and clinical features of drug naive patients with de novo PD and RBD_annotated_ann.sum","Avila et al., (2011) - Effect of temporal lobe structure volume on memory in eldery depressed patients_annotated_ann.sum","Bartova et al., (2010) - Correlation between substantia nigra features detected by sonography and PD_annotated_ann.sum","Benninger et al., (2008) - Morphological differences in PD with and without rest tremor_annotated_ann.sum","Beyer et al., (2006) - MRI study of PD with MCI and dementia using VBM_annotated_ann.sum","Bilello et al., (2015) - Correlating cognitive decline with white matter lesions and atrophy in AD_annotated_ann.sum","Birn et al., (2010) - Neural systems supporting lexical search guided by letter and semantic category cues. A self-paced overt response fMRI study of verbal fluency._annotated_ann.sum","Biundo et al., (2015) - Patterns of cortical thickness associated with ICD in PD_annotated_ann.sum","Bologna et al., (2016) - Neural correlates of blinking abnormalities in patients with progressive supranuclear palsy_annotated_ann.sum","Brandi et al., (2014) - The neural correlates of planning and executing actual tool use_annotated_ann.sum","Brugnolo et al., (2014) - Metabolic correlates of rey auditory verbal learning test in elderly subjects with memory complaints_annotated_ann.sum","Clark et al., (2014) - Lexical factors and cerebral regions influcing verbal fluency performance in MCI_annotated_ann.sum","Cotelli et al., (2010) - Action and object naming in physiological aging. An rTMS study_annotated_ann.sum","Coull et al., (1996) - A fronto-parietal network for RVIP_annotated_ann.sum"];
	$(document).ready(function(){
		for(var i=0;i<files.length;i++){
			$('#file_list').append($("<option></option>")
	                    .attr("value",files[i])
	                    .text(decodeURIComponent(files[i]))); 
		}

		$('#file_list').on('change', function() {
			tmining.load_text(this.value);
		});
		$('#file_list option:eq(0)').prop('selected', true);
		tmining.load_text($("#file_list option:selected").text());
	})

})(this.jQuery)