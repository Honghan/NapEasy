/**K-Drive project: query generation demo: UI //KT Group@UNIABDN*/
(function($) {
	if(typeof tmining == "undefined") {
		tmining = {
			server_prefix: "http://honghan.info/pn-tmining/summaries/",

			load_text: function(file_path){
				var url = tmining.server_prefix + file_path
				console.log(url)
				$('#loadDiv').load(url, function(){
					alert('done loading.')
				});
			},

			load_script: function(file_path, method, sendData, success, error){
				var url = tmining.server_prefix + file_path
				console.log(url)
				$.ajax({
					   type: method || "Get",
					   url: url,
					   data: sendData || [],
					   cache: false,
					   dataType: "html", /* use "html" for HTML, use "json" for non-HTML */
					   success: success | tmining.load_cb /* (data, textStatus, jqXHR) */ || null,
					   error: error /* (jqXHR, textStatus, errorThrown) */ || null
				});
			},

			load_cb:function(data, textStatus, jqXHR){
				console.log("callback...")
				console.log(data)
			}
		}
	}
})(jQuery);