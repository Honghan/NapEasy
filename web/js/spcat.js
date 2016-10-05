(function($){
    function renderCats(){
        var s = "<button class='btn'>JSON it</button>";
        for (var i=0;i<sp_patterns.length;i++){
            s += "<div class='pattern'>";

            s += "<div class='pattern-content label'>";
            s += (sp_patterns[i][0].s ? sp_patterns[i][0].s.join("-") : "") + " " +
                (sp_patterns[i][0].p ? sp_patterns[i][0].p.join("-") : "");
            s += "</div>";

            s += "<div class='pattern-content'>";
            for (var cat in sp_cats){
                var idx = sp_cats[cat].indexOf("" + i);
                s += "<input type='checkbox' index='" + i + "' name='" + cat + "' " +
                    (idx >= 0 ? "checked" : "") + "/>" + cat + " ";
            }
            s += "</div>";

            s += "</div>";
        }
        s += "<button class='btn'>JSON it</button>";
        $('#sp').html(s);
    }

	$(document).ready(function(){
		renderCats();
		$('.btn').click(function(){
		    var ret = {};
            $( "input:checked" ).each(function(){
                var cat = $(this).attr('name');
                if (cat in ret)
                    ret[cat].push($(this).attr('index'));
                else
                    ret[cat] = [$(this).attr('index')];
            });
            $('#catJson').val($.toJSON(ret));
            $('#catJson').show();
		});
	})

})(this.jQuery)