/**K-Drive project: query generation demo: UI //KT Group@UNIABDN*/
(function($) {
	if(typeof tmining == "undefined") {
		tmining = {
			server_prefix: "http://honghan.info/pn-tmining/summaries/",
			cur_file: "",
			load_text: function(file_path){
				tmining.cur_file = file_path;
				var url = tmining.server_prefix + encodeURIComponent(file_path);
				console.log(url)
				$('#loadDiv').load(url, function(){
					eval("var summ = " + $('#loadDiv').html());
					console.log(summ);
					tmining.render_summary(summ);
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
			},

			render_summary: function(summ){
				$('.title').html(decodeURIComponent(tmining.cur_file));
				var s = '';
				if (summ.hasOwnProperty('goal'))
					s += tmining.render_typed_sentence('goal', summ.goal);
				if (summ.hasOwnProperty('method'))
					s += tmining.render_typed_sentence('method', summ.method);
				if (summ.hasOwnProperty('findings'))
					s += tmining.render_typed_sentence('findings', summ.findings);
				if (summ.hasOwnProperty('general'))
					s += tmining.render_typed_sentence('general', summ.general);
				$('.summary').html(s);
			},

			render_typed_sentence: function(type, sentences){
				var s = "<div class='stype'>" + type + "</div>";
				for(var i=0;i<sentences.length;i++)
					s += "<div class='sentence'><span class='sid'>" + sentences[i][1].sid + ".</span>" + sentences[i][0] + "</div>"
				return s;
			}
		}
	}
})(jQuery);